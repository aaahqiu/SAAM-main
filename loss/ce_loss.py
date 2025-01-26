import torch.nn as nn

def CE_loss(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)