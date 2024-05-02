import os
import sys
import logging
import random
import numpy as np
import torch
import torch.distributed as dist


def init_logging(log_root, models_root=None):
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    if models_root is not None:
        handler_file = logging.FileHandler(
            os.path.join(models_root, "training.log"))
        handler_file.setFormatter(formatter)
        log_root.addHandler(handler_file)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_stream)


def distributed_init(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.distributed = True
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.world_size, rank=args.rank)
        # dist.barrier()
    else:
        args.rank = 0
        args.world_size = 1
        args.distributed = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    device = length.device if device is None else device
    mask = torch.arange(max_len,
                        device=device, dtype=length.dtype).expand(
                            len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask
