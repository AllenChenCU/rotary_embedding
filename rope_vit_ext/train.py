import time
import os

import structlog
import torch
from torch.optim import lr_scheduler

from utils import AverageMeter, ProgressMeter, save_on_master


logger = structlog.get_logger()


def train(
    args,
    trainloader, 
    testloader, 
    net, 
    criterion, 
    optimizer, 
    lr_scheduler=None, 
    scaler=None,
    num_epochs=5, 
    train_metrics={}, 
    test_metrics={}, 
):
    best_acc = 0

    for epoch in range(num_epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
        train_result = train_one_epoch(epoch, trainloader, net, criterion, optimizer, args, scaler)
        test_result, best_acc = test_one_epoch(epoch, testloader, net, criterion, args, best_acc)
        if args.distributed:
            train_metrics[str(epoch)][str(args.rank)] = train_result
            test_metrics[str(epoch)][str(args.rank)] = test_result
        else:
            train_metrics[str(epoch)] = train_result
            test_metrics[str(epoch)] = test_result
        if lr_scheduler:
            lr_scheduler.step()
    return train_metrics, test_metrics


def train_one_epoch(epoch, dataloader, net, criterion, optimizer, args, scaler=None):
    logger.info(f'Training epoch: {epoch} \n')

    # Config
    training_time = AverageMeter('training_time', ':6.4f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [training_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch)
    )
    results = {}
    net.train()

    # Loop thru batches to train
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        # train
        if args.distributed:
            dest = args.rank
        else:
            dest = args.device
        inputs, targets = inputs.to(dest), targets.to(dest)
        if args.device == "cuda":
            torch.cuda.synchronize()
        training_start = time.perf_counter()
        optimizer.zero_grad()
        outputs = net(inputs)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if args.device == "cuda":
            torch.cuda.synchronize()
        curr_time_training = time.perf_counter()
        training_time.update(curr_time_training - training_start)

        # Metrics
        # Calculate losses
        batch_loss = loss.item()
        losses.update(batch_loss, inputs.size(0))
        # Calculate accuracies
        _, predicted = outputs.max(1)
        batch_total = targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()
        batch_top1 = batch_correct / batch_total
        top1.update(batch_top1, batch_total)

        # Gather results
        result_batch = {
            # batch level
            "BatchLoss": batch_loss, 
            "BatchTotal": batch_total, 
            "BatchCorrect": batch_correct, 
            "BatchAcc@1": batch_top1, 
            # epoch level
            "Loss": losses.avg, 
            "Total": top1.count, 
            "Correct": top1.sum, 
            "Acc@1": top1.avg, 
            # time
            "BatchTrainingTime": training_time.val, 
            "TrainingTime": training_time.sum, 
        }
        results[str(batch_idx)] = result_batch

        # print progress
        progress.display(batch_idx)

    return results


def test_one_epoch(epoch, dataloader, net, criterion, args, best_acc=0):
    logger.info(f"Testing epoch: {epoch}\n")

    # Config
    inference_time = AverageMeter('inference_time', ':6.4f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [inference_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch)
    )
    net.eval()

    # Loop thru batches to test
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            # Compute
            if args.distributed:
                dest = args.rank
            else:
                dest = args.device
            inputs, targets = inputs.to(dest), targets.to(dest)
            if args.device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if args.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            inference_time.update(end_time - start_time)

            # Metrics
            # Calculate Losses
            batch_loss = loss.item()
            losses.update(batch_loss, inputs.size(0))
            # Calculate Accuracies
            _, predicted = outputs.max(1)
            batch_total = targets.size(0)
            batch_correct = predicted.eq(targets).sum().item()
            batch_top1 = batch_correct / batch_total
            top1.update(batch_top1, batch_total)

            # print progress
            progress.display(batch_idx)

    # Save checkpoint.
    acc = top1.avg
    if acc > best_acc:
        logger.info(f'Saving model for epoch {epoch} with best acc at {acc}...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        curr_dir = os.path.dirname(__file__)
        save_folder = os.path.join(curr_dir, "model_registry", args.run_id)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, f"{args.run_id}.pth")
        save_on_master(state, save_path)
        best_acc = acc
    
    return {
        # Metrics
        "Loss": losses.avg, 
        "Total": top1.count, 
        "Correct": top1.sum, 
        "Acc@1": top1.avg, 
        # Time
        "InferenceTime": inference_time.sum, 
    }, best_acc
