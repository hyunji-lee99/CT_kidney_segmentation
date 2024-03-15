import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from utils.postprocess import colour_code_segmentation

def print_segmentation_output(dataset, best_model, device):
    for i in range(10):
        idx = np.random.randint(0, len(dataset))
        image, label = dataset[idx]

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

        # Predict test image
        pred_mask = best_model(x_tensor)
        # pred_mask=test_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # x=np.argmax(pred_mask, axis=0)

        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Convert image, label from `CHW` format to `HWC` format
        image = np.transpose(image, (1, 2, 0))
        label = np.transpose(label, (1, 2, 0))

        # 출력 전 이미지 dimension 변경
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('original')
        plt.imshow(image, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title('ground-truth')
        plt.imshow(colour_code_segmentation(np.argmax(label, axis=2)))

        plt.subplot(1, 3, 3)
        plt.title("prediction")
        plt.imshow(colour_code_segmentation(np.argmax(pred_mask, axis=2)))

        plt.show()

def print_logs(train_logs_list, valid_logs_list, score_name):
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.transpose()
    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.score_name.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.score_name.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=21)
    plt.ylabel(f'{score_name} Score', fontsize=21)
    plt.ylim([-0.5, 1.5])
    plt.title(f'{score_name} Score Plot', fontsize=21)
    plt.grid()
    plt.savefig(f'result/{score_name}_score_plot.png')
    plt.show()