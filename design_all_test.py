import argparse
import subprocess

def run_design(testset_list, config, out_root, tag, seed, device, batch_size):
    for index in testset_list:
        print(f"processing testset {index}...")
        cmd = [
            "python",
            "design_testset.py",
            str(index),
            "--config", str(config),
            "--out_root", str(out_root),
            "--tag", str(tag),
            "--seed", str(seed),
            "--device", str(device),
            "--batch_size", str(batch_size)
        ]

        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            print(f"命令 {' '.join(cmd)} 执行成功。")
        except subprocess.CalledProcessError as e:
            print(f"命令 {' '.join(cmd)} 执行失败，错误信息: {e.stderr}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_list', type=int, nargs='+', default=60)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()

    if len(args.data_list) == 1:
        data_list = [i for i in range(args.data_list[0])]
    elif len(args.data_list) == 2:
        data_list = [i for i in range(args.data_list[0], args.data_list[1])]
    else:
        data_list = args.data_list

    run_design(
        testset_list=data_list,
        config=args.config,
        out_root=args.out_root,
        tag=args.tag,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
