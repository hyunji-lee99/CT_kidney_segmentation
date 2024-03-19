## Abdominal CT based Axial 2D Kidney Segmentation Pytorch : Implementation of U-Net


![화면 캡처 2024-03-15 114217](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/6040156a-9497-4e13-8449-895619e57f0e)

Abdominal CT Axial based left kidney, right kidney segmentaion Unet model.

*This is my first computer vision deep learning project.* 

*Please let me know if you find my mistake!* 

*Also, if you want to advise me, please contact me.*

*contact : hyunji0483@knu.ac.kr*

## Dataset (KiTS 23)


![Untitled](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/b21dfca1-d6ac-4e26-9c43-e0bc54e3c004) |![Untitled1](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/10c29ffd-31fe-4ee9-bd2f-b09c10672a5c)
--- | --- | 
https://kits-challenge.org/kits23/

**Installation**

```
git clone https://github.com/neheller/kits23
cd kits23
pip3 install -e .
```

**Data Download**

```
kits23_download_data
```

**Original Class** : background(0), kidney(1), tumor(2), cyst(3)
**changed to background(0), left kidney(1), right kidney(2)**

## Model Outputs


![prediction_1](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/61b71228-2382-4f35-9280-35d9e114e7fe)
![prediction_2](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/c7dfab44-168c-4e3a-8ab3-bf2acc47d459)
![prediction_7](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/e2dc3cdf-12ca-464d-a0a1-26279b93013f)
![prediction_9](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/5dcc2529-a46c-4756-9e1a-27fea31022a1)


### Metrics
![Dice Loss_score_plot](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/0b128d4b-5eee-4ea7-86e3-7d1dc4720932)
**Dice Loss**

![IoU_score_plot](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/6b753c96-d3c7-4ef6-a30c-6df8dc5ab299)
**IoU(Intersection over Union)**

## Experiments Table


Exp 1. optimizer = Adam / lr = 0.0001 / loss function = Dice Loss / Batch Size =16 / Epoch = 100

Exp 2. optimizer = Adam / lr = 0.0001 / loss function = Dice CE Loss / Batch Size =16 / Epoch = 100

Exp 3. optimizer = AdamW / lr = 0.0001 / loss function = Dice Loss / Batch Size =16 / Epoch = 100

Exp 4. optimizer = AdamW / lr = 0.0001 / loss function = Dice CE Loss / Batch Size =16 / Epoch = 100

**All Experiments ignore background in calculating loss.**

![화면 캡처 2024-03-19 165253](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/ff7ef83d-88c9-45e4-abf5-851fe60c541e)
