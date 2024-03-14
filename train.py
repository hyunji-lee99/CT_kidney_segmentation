import torch
from tqdm import tqdm

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