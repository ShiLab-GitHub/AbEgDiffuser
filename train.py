import logging
import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from AbEgDiffuser.datasets import get_dataset
from AbEgDiffuser.models import get_model
from AbEgDiffuser.utils.misc import *
from AbEgDiffuser.utils.data import *
from AbEgDiffuser.utils.train import *

import sys
import signal
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

def cleanup(signum, frame):
    print(f"Process {os.getpid()} received signal {signum}, cleaning up...")
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    sys.exit(0)

def main_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, init_method="env://")
    args.rank = rank
    args.world_size = world_size
    args.device = f'cuda:{args.local_rank}'
    torch.cuda.set_device(args.local_rank)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed + rank)

    if args.rank == 0:
        # Logging
        if args.debug:
            logger = get_logger('train', None)
            writer = BlackHole()
        else:
            if args.resume:
                log_dir = os.path.dirname(os.path.dirname(args.resume))
            else:
                log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
            ckpt_dir = os.path.join(log_dir, 'checkpoints')
            if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
            logger = get_logger('train', log_dir)
            writer = torch.utils.tensorboard.SummaryWriter(log_dir)
            tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
            if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
                shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        logger.info(args)
        logger.info(config)
    else:
        logger = BlackHole()
        writer = BlackHole()
    if args.world_size > 1:
        dist.barrier()

    # Data
    if args.rank == 0:
        logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size // world_size,
        sampler=train_sampler,
        collate_fn=PaddingCollate(),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size // world_size,
        sampler=val_sampler,
        collate_fn=PaddingCollate(),
        num_workers=args.num_workers,
        pin_memory=True, 
        persistent_workers=True
        )

    iter_per_epoch = len(train_loader)
    if iter_per_epoch == 0:
        iter_per_epoch = 1

    if args.rank == 0:
        logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    if args.rank == 0:
        logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    if args.world_size > 1:
        # model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank) # , find_unused_parameters=True
    if args.rank == 0:
        logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        if args.rank == 0:
            logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration'] # + 1
        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
        if args.rank == 0:
            logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.rank == 0:
            logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    scaler = GradScaler(enabled=config.train.use_amp)
    # Train
    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

        # Forward
        if config.train.use_amp:
            with autocast(dtype=torch.float16):
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss
        else:
            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
        time_forward_end = current_milli_time()

        # Backward
        optimizer.zero_grad()
        
        if config.train.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        if config.train.use_amp:
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()

        time_backward_end = current_milli_time()

        # Logging
        if args.rank == 0:
            if config.train.use_amp:
                log_losses(loss_dict, it, 'train', logger, writer, others={
                    'grad': orig_grad_norm,
                    'lr': optimizer.param_groups[0]['lr'],
                    'time_forward': (time_forward_end - time_start) / 1000,
                    'time_backward': (time_backward_end - time_forward_end) / 1000,
                    'scale': scaler.get_scale()
                })
            else:
                log_losses(loss_dict, it, 'train', logger, writer, others={
                    'grad': orig_grad_norm,
                    'lr': optimizer.param_groups[0]['lr'],
                    'time_forward': (time_forward_end - time_start) / 1000,
                    'time_backward': (time_backward_end - time_forward_end) / 1000,
                })

            if not torch.isfinite(loss):
                logger.error('NaN or Inf detected.')
                if isinstance(model, DDP):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save({
                    'config': config,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'batch': recursive_to(batch, 'cpu'),
                }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
                raise KeyboardInterrupt()

    # Validate
    def validate(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)
                # Forward
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss

                batch_size = batch['aa'].shape[0]
                loss_tape.update(loss_dict, batch_size)

        avg_loss = loss_tape.log(args, it, logger, writer, 'val')

        if config.train.scheduler.type == 'plateau':
            if args.rank == 0:
                scheduler.step(avg_loss)
            else:
                scheduler.step(0.0)
        else:
            if args.rank == 0:
                scheduler.step()

        if args.world_size > 1:
            scheduler_state = scheduler.state_dict()
            dist.broadcast_object_list([scheduler_state], src=0)
            if args.rank != 0:
                scheduler.load_state_dict(scheduler_state)
        return avg_loss

    try:
        Best_Loss = float('inf')
        Best_Iter = 0
        early_stop_count = 0

        for it in range(it_first, config.train.max_iters + 1):
            if train_sampler is not None:
                epoch = it // iter_per_epoch
                train_sampler.set_epoch(epoch)
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)

                if (Best_Loss > avg_val_loss):
                    Best_Loss = avg_val_loss
                    Best_Iter = it
                    early_stop_count = 0
                    if args.rank == 0 and not args.debug:
                    # if not args.debug:
                        # ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                        if isinstance(model, DDP):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()
                        ckpt_path = os.path.join(ckpt_dir, 'Best_Ab.pt')
                        torch.save({
                            'config': config,
                            'model': state_dict,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                else:
                    early_stop_count += 1

                if args.rank == 0 and not args.debug:
                    if isinstance(model, DDP):
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    ckpt_path = os.path.join(ckpt_dir, 'current_Ab.pt')
                    torch.save({
                        'config': config,
                        'model': state_dict,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)

                logger.info('Best_Iter={}; Best_Loss={:.4f}; early_stop_count = {}.'.format(Best_Iter, Best_Loss,
                                                                                            early_stop_count))
                if early_stop_count >= 30:
                    logger.info('Early stopping triggered')
                    break

    except KeyboardInterrupt:
        cleanup(None, None)
        logger.info('Terminating...')

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py ./configs/train/codesign_single.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py ./configs/train/codesign_single.yml
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py ./configs/train/codesign_single.yml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    args = parser.parse_args()

    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    main_worker(args.local_rank, torch.cuda.device_count(), args)

