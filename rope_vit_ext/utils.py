import os

import torch
import torch.distributed as dist

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: Batch - {val' + self.fmt + '} | Epoch - {avg' + self.fmt + '} ({sum}/{count}) \n'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']\n'


# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__
#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print


def init_distributed_mode(args, rank, world_size):
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ['WORLD_SIZE'])
    #     args.gpu = int(os.environ['LOCAL_RANK'])
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # else:
    #     print('Not using distributed mode')
    #     args.distributed = False
    #     return

    #args.distributed = True
    args.rank = rank
    args.gpu = rank % torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    #print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist_url = os.getenv("MASTER_ADDR") + ":" + os.getenv("MASTER_PORT")
    print('| distributed init (rank {}): {}'.format(args.rank, dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, #init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    #setup_for_distributed(args.rank == 0)


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
