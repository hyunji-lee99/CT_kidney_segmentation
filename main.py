import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.save_dataset import save_dataset
import numpy as np
from data_loader.dataloader import SegmentationDataset
from torch.utils.data import DataLoader
from models.UNet import UNet
import torch
import gc
from train import train_fn
from eval import eval_fn
from losses.SoftDiceLoss import SoftDiceLoss
from losses.DiceLoss import DiceLoss
from inference import print_segmentation_output, print_logs

# load kits23 dataset
DATA_DIR = './data/kits23/dataset/'

ORIGINAL_IMAGES = []
MASK_IMAGES = []

CASES = sorted(os.listdir(DATA_DIR))
for c in CASES:
    if c == 'kits23.json':
        continue
    if len(os.listdir(DATA_DIR+c)) == 2:
        ORIGINAL_IMAGES.append(os.path.join(DATA_DIR, c, "imaging.nii.gz"))
        MASK_IMAGES.append(os.path.join(DATA_DIR, c, "segmentation.nii.gz"))

df_data = pd.DataFrame({'image': ORIGINAL_IMAGES, 'label': MASK_IMAGES})
print(f"Number of Data : {len(df_data)}")

# reduced the number of patients to 300 due to lack of memory issue
# split train set / valid set
train_df, valid_df = train_test_split(df_data[:300], test_size=0.2, random_state=42)
print(f"number of train set : {len(train_df)}, number of valid set : {len(valid_df)}")

# save compressed dataset to npz file
save_dataset(train_df, input='train')
save_dataset(valid_df, input='valid')

# load saved data
# train set
concat_train_image_npz = np.load("data/train_image_concat.npz")['data'].astype(np.float32)/255.0
concat_train_label_npz = np.load("data/train_label_concat.npz")['data'].astype(np.float32)
concat_train_index_npz = np.load("data/train_index_concat.npz")['data']
# valid set
concat_valid_image_npz = np.load("data/valid_image_concat.npz")['data'].astype(np.float32)/255.0
concat_valid_label_npz = np.load("data/valid_label_concat.npz")['data'].astype(np.float32)
concat_valid_index_npz = np.load("data/valid_index_concat.npz")['data']

train_set = SegmentationDataset(data_image_array=concat_train_image_npz, data_label_array=concat_train_label_npz,
                                data_idx_array=concat_train_index_npz, augmentations=None)
valid_set = SegmentationDataset(data_image_array=concat_valid_image_npz, data_label_array=concat_valid_label_npz,
                                data_idx_array=concat_valid_index_npz, augmentations=None)

# set hyperparameter
DEVICE = 'cuda:3'
# Set num of epochs
EPOCHS = 100
model = UNet().to(DEVICE)
# GPU 번호 지정(0~3) cuda:0, 1 등등..
# define optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=1e-3),
])

optimizer_w = torch.optim.AdamW([
    dict(params=model.parameters(), lr=1e-3),
])

# define data loader
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
print(f"total number of batches in train loader : {len(train_loader)}")
print(f"total number of batches in valid loader : {len(valid_loader)}")

# empty cache for preventing cuda out of memory issue
torch.cuda.empty_cache()
gc.collect()

# train model
train_logs_list, valid_logs_list = [], []
best_valid_loss = np.inf

for i in range(EPOCHS):
    train_loss, train_iou = train_fn(train_loader, model, optimizer, DiceLoss, DEVICE)
    valid_loss, valid_iou = eval_fn(valid_loader, model, DiceLoss, DEVICE)
    train_logs_list.append({'Dice Loss': train_loss, 'IoU': train_iou})
    valid_logs_list.append({'Dice Loss': valid_loss, 'IoU': valid_iou})

    if valid_loss < best_valid_loss:
        torch.save(model, './SavedModel/best_model.pt')
        print('Model Saved')
        best_valid_loss = valid_loss

    print(
        f"EPOCH : {i + 1} Train Loss : {train_loss} Valid Loss : {valid_loss}"
        f"Train IoU : {train_iou} Valid IoU : {valid_iou}")

# inference
best_model = torch.load('./SavedModel/best_model.pt')
print_segmentation_output(valid_set, best_model, DEVICE)
print_logs(train_logs_list, valid_logs_list, score_name='IoU')
print_logs(train_logs_list, valid_logs_list, score_name='Dice Loss')