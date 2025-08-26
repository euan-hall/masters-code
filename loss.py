import torch.nn as nn

class ContrastiveLearningLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLearningLoss, self).__init__()

    def forward(self, z_a, z_b):
        n = len(z_a)

        alignment = 0
        for i in range(n):
            alignment += -torch.sum(torch.dot(z_a[i].T, z_b[i]))
        uniformity_loss = 0
        for i in range(n):
            negative_sum = 0
            for j in range(n-1):
                if i == j:
                    continue
                negative_sum += torch.sum(torch.dot(z_a[i].T, z_b[i]))
            
            uniformity = torch.exp(torch.dot(z_a[i].T, z_b[i]))
            uniformity_loss += negative_sum + uniformity
        return alignment + uniformity_loss

class InstanceDiscriminationLoss(nn.Module):
    def __init__(self):
        super(InstanceDiscriminationLoss, self).__init__()

    def forward(self, predictions):
        return -torch.sum(torch.log(predictions))