# MIT License
# Copyright (c) 2026 MULAN authors

import os
import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier(device_ids=[args.gpu])
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if get_world_size() == 1:
        return tensor

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def concat_all_gather_rewrite(tensor):
    """
    Performs all_gather operation on the provided tensor and overwrites the input to recover local gradients.

    Args:
        tensor (torch.Tensor): A tensor of arbitrary shape to be gathered across all processes.

    Returns:
        torch.Tensor: A tensor containing the gathered tensors from all processes, concatenated along the first dimension.
        If world size is 1, returns the input tensor unchanged.
    """
    if get_world_size() == 1:
        return tensor

    tensors_gather = [torch.empty_like(tensor) for _ in range(get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    rank = get_rank()
    tensors_gather[rank] = tensor  # overwrite to recover local gradients

    output = torch.cat(tensors_gather, dim=0)
    return output
