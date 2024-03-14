import torch.nn as nn
from losses import SoftDiceLoss

def DiceCELoss(predicted, target, lambda_dice=1.0, lambda_ce=1.0):
    diceloss, iou = SoftDiceLoss(predicted, target)

    # softmax가 cross entropy 내부에서 발생하기 때문에, 기존 dice loss는 세 가지 클래스에 대한 softmax를 수행하는 반면에 cross entropy는
    # 2가지 클래스에 대해서만 softmax를 수행하므로 diceceloss를 사용해도 loss는 오히려 더 줄어들 수 있다.
    # 수정 요망

    # cross entropy의 reduction default는 mean
    # 즉, 다른 설정을 해주지 않으면 기본적으로 평균치를 내서 단일 스칼라값이 출력됨.
    # cross_entropy=nn.CrossEntropyLoss(ignore_index=0)
    # ce=cross_entropy(predicted,torch.argmax(target, axis=1))

    # 이 loss가 제대로 값이 나오지 않을시, onehot 값에서 class 0에 해당하는 target[:,0]=0으로 주고 gradient되지 않도록 설정

    cross_entropy = nn.CrossEntropyLoss()
    # ignore background
    # target[:,0]=0.0
    # 얘 때문에 RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.DoubleTensor [8, 2, 512, 512]] is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True). 에러 발생함. 이유는 노션 정리해둠

    # cross_entropy=nn.CrossEntropyLoss(weight=weights)

    ce = cross_entropy(predicted[:, 1:], target[:, 1:])
    loss = ce * lambda_ce + diceloss * lambda_dice
    return loss, iou