import torch
from tqdm import tqdm

def eval_fn(loader, model, criterion, device):
    model.eval()
    total_loss =0.0
    total_iou =0.0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images =images.to(device)
            labels =labels.to(device)

            output = model(images)
            # loss, iou = dice_loss(output, labels)
            loss, iou = criterion(output, labels)

            total_loss+=loss.item()
            total_iou+=iou.item()

    return total_loss /len(loader), total_iou /len(loader)