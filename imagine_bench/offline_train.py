import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='bc', help='Algorithm to use')
    parser.add_argument('--env', default='Mujoco-v0', help='Environment to use')
    parser.add_argument('--ds_type', default='rephrase', help='Dataset level')
    parser.add_argument("--device", type=str, default="cuda:0", help="The device for offlineRL training.")
    parser.add_argument("--seed", type=int, default=7, help="Seed.")
    parser.add_argument("--eval_episodes", type=int, default=10)
    args = parser.parse_args()
    if args.env == 'Ball-v0':
        cmd = ['python', 'train1.py', '--algo', args.algo, '--env', args.env, '--ds_type', args.ds_type, '--device', args.device, '--seed', args.seed, '--eval_episodes', args.eval_episodes]
    elif args.env == 'MetaWorld-v0':
        cmd = ['python', 'train1.py', '--algo', args.algo, '--env', args.env, '--ds_type', args.ds_type, '--device', args.device, '--seed', args.seed, '--eval_episodes', args.eval_episodes]
    elif args.env == 'BabyAI-v0':
        cmd = ['python', 'train1.py', '--algo', args.algo, '--env', args.env, '--ds_type', args.ds_type, '--device', args.device, '--seed', args.seed, '--eval_episodes', args.eval_episodes]
    elif args.env == 'Libero-v0':
        cmd = ['python', 'train.py', '--algo', args.algo, '--env', args.env, '--ds_type', args.ds_type, '--device', args.device, '--seed', args.seed, '--eval_episodes', args.eval_episodes]
    elif args.env == 'Mujoco-v0':
        cmd = ['python', 'train.py', '--algo', args.algo, '--env', args.env, '--ds_type', args.ds_type, '--device', args.device, '--seed', args.seed, '--eval_episodes', args.eval_episodes]

    # 调用对应的脚本
    subprocess.run(cmd)

if __name__ == '__main__':
    main()