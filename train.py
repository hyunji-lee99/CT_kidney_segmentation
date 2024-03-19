import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc
import numpy as np
from models.UNet import UNet
from losses.SoftDiceLoss import SoftDiceLoss
from losses.DiceLoss import DiceLoss
from losses.DiceCELoss import DiceCELoss

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()


def train_fn(loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(images)
            loss, iou = criterion(output, labels)
        # loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        total_loss += loss.item()
        total_iou += iou.item()

    return total_loss / len(loader), total_iou / len(loader)


def eval_fn(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images =images.to(device)
            labels =labels.to(device)

            output = model(images)
            # loss, iou = dice_loss(output, labels)
            loss, iou = criterion(output, labels)

            total_loss += loss.item()
            total_iou += iou.item()

    return total_loss / len(loader), total_iou / len(loader)

# set hyperparameter
EPOCHS = 100
# Select GPU number ex) cuda:0, 1 etc..
DEVICE = 'cuda:3'


def train(train_loader, valid_loader, opt, loss, exp_num):
    # model init
    model = UNet().to(DEVICE)
    # define optimizer
    if opt == 'Adam':
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=1e-3),
        ])
    elif opt == 'AdamW':
        optimizer = torch.optim.AdamW([
            dict(params=model.parameters(), lr=1e-3),
        ])

    # define loss function
    if loss == 'DiceLoss':
        loss_fn = DiceLoss
    elif loss == 'DiceCELoss':
        loss_fn = DiceCELoss
    elif loss == 'SoftDiceLoss':
        loss_fn = SoftDiceLoss

    # for tensorboard
    writer = SummaryWriter()
    # empty cache for preventing cuda out of memory issue
    torch.cuda.empty_cache()
    gc.collect()

    # train model
    train_logs_list, valid_logs_list = [], []
    best_valid_loss = np.inf

    for i in range(EPOCHS):
        train_loss, train_iou = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        valid_loss, valid_iou = eval_fn(valid_loader, model, loss_fn, DEVICE)
        train_logs_list.append({'Loss': train_loss, 'IoU': train_iou})
        valid_logs_list.append({'Loss': valid_loss, 'IoU': valid_iou})

        # tensorboard
        writer.add_scalar(f"{exp_num}/Loss/train", train_loss, i)
        writer.add_scalar(f"{exp_num}/Loss/valid", valid_loss, i)
        writer.add_scalar(f"{exp_num}/IoU/train", train_iou, i)
        writer.add_scalar(f"{exp_num}/IoU/valid", valid_iou, i)

        if valid_loss < best_valid_loss:
            torch.save(model, f'./SavedModel/best_model_exp{exp_num}.pt')
            print('Model Saved')
            best_valid_loss = valid_loss

        print(
            f"EPOCH : {i + 1} Train Loss : {train_loss} Valid Loss : {valid_loss}"
            f"Train IoU : {train_iou} Valid IoU : {valid_iou}")

    writer.flush()
    writer.close()

    return train_logs_list, valid_logs_list

