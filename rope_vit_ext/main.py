import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#from torchvision import models
import pandas as pd
import structlog
from timm.models import create_model
import torch.distributed as dist
import torch.multiprocessing as mp

from data import build_dataset
from train import train
import models
from utils import (
    init_distributed_mode, get_world_size, get_rank
)


logger = structlog.get_logger()


def main(rank, world_size, args, train_metrics, test_metrics):

    logger.info(f"Distributed Training: {args.distributed}")
    logger.info(f"Pass in rank: {rank}")
    if args.distributed:
        init_distributed_mode(args, rank, world_size)
        logger.info(f"World size: {get_world_size()}")
        logger.info(f"Rank: {get_rank()}")

    # Data
    logger.info(f"Dataset: {args.dataset}")
    dataset_train, num_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    trainloader = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True, 
        drop_last=True,
    )
    testloader = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    logger.info(f"Model: {args.model}")
    model = create_model(
        args.model, 
        pretrained=args.pretrained,
        num_classes=num_classes,
        drop_rate=0.0, 
        drop_path_rate=0.1, 
        drop_block_rate=None, 
        #img_size=224, 
    )
    model = model.to(args.device)
    model_without_ddp = model
    if args.distributed:
        model = model.to(device=rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * get_world_size() / 512.0
        args.lr = linear_scaled_lr

    # Train Config
    logger.info(f"Training...")
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.AdamW(model_without_ddp.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    num_epochs = args.epochs

    # train
    train_metrics, test_metrics = train(
        args, trainloader, testloader, model, criterion, optimizer, lr_scheduler, num_epochs, 
        train_metrics, test_metrics,
    )

    if args.distributed:
        dist.destroy_process_group()

    # save
    logger.info(f"Saving...")
    if args.distributed:
        train_metrics_df1 = pd.DataFrame.from_dict(
            {
                (int(i), int(j), int(k)): train_metrics[i][j][k] for i in train_metrics.keys() for j in train_metrics[i].keys() for k in train_metrics[i][j].keys()
            }, orient="index", 
        )
        train_metrics_df2 = (
            train_metrics_df1
            .reset_index(names=["epoch_num", "rank_num", "batch_num"])
            .groupby(["epoch_num", "rank_num"])
            .last()
            [["Loss", "Total", "Correct", "Acc@1", "TrainingTime"]]
        )
        test_metrics_df = pd.DataFrame.from_dict(
            {
                (int(i), int(j)): test_metrics[i][j] for i in test_metrics.keys() for j in test_metrics[i].keys()
            }, orient="index", 
        )
    else:
        train_metrics_df1 = pd.DataFrame.from_dict(
            {
                (int(i), int(j)): train_metrics[i][j] for i in train_metrics.keys() for j in train_metrics[i].keys()
            }, orient="index",
        )
        train_metrics_df2 = (
            train_metrics_df1
            .reset_index(names=["epoch_num", "batch_num"])
            .groupby("epoch_num")
            .last()
            [["Loss", "Total", "Correct", "Acc@1", "TrainingTime"]]
        )
        test_metrics_df = pd.DataFrame.from_dict(test_metrics, orient="index")

    if not os.path.isdir(args.run_id):
        os.mkdir(args.run_id)
    train_metrics_batch_filepath = os.path.join(args.run_id, args.run_id + "_metrics_train_batch.csv")
    train_metrics_epoch_filepath = os.path.join(args.run_id, args.run_id + "_metrics_train_epoch.csv")
    test_metrics_epoch_filepath = os.path.join(args.run_id, args.run_id + "_metrics_test_epoch.csv")
    train_metrics_df1.to_csv(train_metrics_batch_filepath)
    train_metrics_df2.to_csv(train_metrics_epoch_filepath)
    test_metrics_df.to_csv(test_metrics_epoch_filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train and evaluate parser")
    parser.add_argument("--cuda", action="store_true", default=False, help="use of cuda")
    parser.add_argument("--data_path", default="./data", type=str, help="file path for all data")
    parser.add_argument("--optimizer", default="sgd", type=str, help="Optimizer")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs for training")
    parser.add_argument("--run_id", default="test_run", type=str, help="run id for naming the metrics files")
    parser.add_argument("--model", default="vit_small_patch16_224", type=str, help="model used")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset used")
    parser.add_argument("--input_size", default=32, type=int, help="desired image input size into the model")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="number of workers for dataloader")
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', help='url used to set up distributed training') #'env://'
    parser.add_argument('--repeated-aug', action='store_true', default=False)
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    #parser.set_defaults(repeated_aug=True)
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--pretrained', action='store_true', default=False, help="whether to load pre-trained model weights")
    args = parser.parse_args()

    # Global Config
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Device: {device}")
    args.device = device

    if args.distributed:
        manager = mp.Manager()
        train_metrics = manager.dict()
        test_metrics = manager.dict()
        for epoch in range(args.epochs):
            train_metrics[str(epoch)] = manager.dict()
            test_metrics[str(epoch)] = manager.dict()
        mp.spawn(
            main, 
            args=(args.world_size, args, train_metrics, test_metrics), 
            nprocs=args.world_size,
        )
    else:
        main(0, 1, args, {}, {})
