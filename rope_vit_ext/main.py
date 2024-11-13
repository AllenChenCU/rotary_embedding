import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import pandas as pd
import structlog

from data import prepare_cifar10
from train import train


logger = structlog.get_logger()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train and evaluate parser")

    # User parser.add_argument() - Documentation : https://docs.python.org/3/library/argparse.html
    parser.add_argument("--cuda", action="store_true", help="use of cuda")
    parser.add_argument("--data_path", default="./data", type=str, help="file path for all data")
    #parser.add_argument("--num_workers", default=2, type=int, help="Number of dataloader workers")
    #parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs for training")
    parser.add_argument("--run_id", default="test_run", type=str, help="run id for naming the metrics files")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Device: {device}")

    # Data
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    batch_size = 64
    trainloader, testloader = prepare_cifar10(batch_size=batch_size)

    # Model
    # base model
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    # Train Config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    num_epochs = args.epochs

    # train
    train_metrics, test_metrics = train(
        trainloader, testloader, model, criterion, optimizer, lr_scheduler, num_epochs, device
    )

    # save
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
        [["Loss", "Total", "Correct", "Acc@1"]]
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


