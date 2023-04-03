import argparse
import os
import time

import torch
import torch.nn as nn


def main():
    # 步骤一：定义local_rank
    parser = argparse.ArgumentParser()
    ...
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()

    # 步骤二：初始化
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # 步骤三：模型分布式处理
    model = nn.Conv2d(2, 2, kernel_size=(3, 3))
    model.to(device)
    num_gpus = torch.cuda.device_count()
    if args.local_rank == 0:
        print("========================================")
    else:
        time.sleep(3)
        print("******************************************")
    print(f'{args.local_rank}')
    print(model.weight)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
    if args.local_rank == 0:
        print("========================================")
    else:
        time.sleep(3)
        print("******************************************")
    print(f'{args.local_rank}')
    print(model.module.weight)


if __name__ == "__main__":
    main()
