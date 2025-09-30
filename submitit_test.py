import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    setup(rank, world_size)

    tensor = torch.ones(1).cuda(rank) * rank
    print(f"Rank {rank} has tensor {tensor.item()}")


    dist.all_reduce(tensor)
    print(f"Rank {rank} after all_reduce has tensor {tensor.item()}")

    cleanup()

def main():
    world_size = 2  
    mp.spawn(demo_basic, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

