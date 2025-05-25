import argparse
import ray
import time
import logging

from AbEgDiffuser.tools.relax.openmm_relaxer import run_openmm
from AbEgDiffuser.tools.relax.pyrosetta_relaxer import run_pyrosetta, run_pyrosetta_fixbb
from AbEgDiffuser.tools.relax.base import TaskScanner


@ray.remote(num_gpus=1/8, num_cpus=1, max_retries=0)
def run_openmm_remote(task):
    try:
        return run_openmm(task)
    except Exception as e:
        task.mark_failure()
        logging.error(f"[SKIPPED] {task.current_path} failed with error: {str(e)}")
        return task


@ray.remote(num_cpus=1, max_retries=0)
def run_pyrosetta_remote(task):
    try:
        return run_pyrosetta(task)
    except Exception as e:
        task.mark_failure()
        logging.error(f"[SKIPPED] {task.current_path} failed with error: {str(e)}")
        return task


@ray.remote(num_cpus=1, max_retries=0)
def run_pyrosetta_fixbb_remote(task):
    try:
        return run_pyrosetta_fixbb(task)
    except Exception as e:
        task.mark_failure()
        logging.error(f"[SKIPPED] {task.current_path} failed with error: {str(e)}")
        return task


@ray.remote
def pipeline_openmm_pyrosetta(task):
    funcs = [
        run_openmm_remote,
        run_pyrosetta_remote,
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)


@ray.remote
def pipeline_pyrosetta(task):
    funcs = [
        run_pyrosetta_remote,
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)


@ray.remote
def pipeline_pyrosetta_fixbb(task):
    funcs = [
        run_pyrosetta_fixbb_remote,
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)


pipeline_dict = {
    'openmm_pyrosetta': pipeline_openmm_pyrosetta,
    'pyrosetta': pipeline_pyrosetta,
    'pyrosetta_fixbb': pipeline_pyrosetta_fixbb,
}


def main():
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pipeline', type=lambda s: pipeline_dict[s], default=pipeline_openmm_pyrosetta)
    args = parser.parse_args()

    final_pfx = 'fixbb' if args.pipeline == pipeline_pyrosetta_fixbb else 'rosetta'
    scanner = TaskScanner(args.root, final_postfix=final_pfx)
    while True:
        tasks = scanner.scan()
        futures = [args.pipeline.remote(t) for t in tasks]
        if len(futures) > 0:
            print(f'Submitted {len(futures)} tasks.')
        while len(futures) > 0:
            done_ids, futures = ray.wait(futures, num_returns=1)
            for done_id in done_ids:
                try:
                    done_task = ray.get(done_id)
                    if done_task.status == 'failed':
                        scanner.record_failure(done_task)
                        print(f"[SKIPPED] Failed task: {done_task.current_path}")
                    else:
                        print(f'Remaining {len(futures)}. Finished {done_task.current_path}')
                except Exception as e:
                    print(f"[WARNING] Task failed with unrecoverable error: {str(e)}")
                # done_task = ray.get(done_id)
                # print(f'Remaining {len(futures)}. Finished {done_task.current_path}')
        time.sleep(1.0)

if __name__ == '__main__':
    main()
