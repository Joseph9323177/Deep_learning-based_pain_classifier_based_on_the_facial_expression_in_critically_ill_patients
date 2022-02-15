import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, arch, pretrained, emb_dim=512):
        super(Encoder, self).__init__()
        if 'resnet' in arch:
            res = torch.hub.load('pytorch/vision:v0.8.0', arch, pretrained=pretrained)
            dim_embedding = 512 if arch=='resnet18' or arch=='resnet34' else 2048
            self.encoder = nn.Sequential(*list(res.children())[:-1], nn.Flatten(), nn.Linear(dim_embedding, emb_dim))
        elif 'vgg' in arch:
            vgg = torch.hub.load('pytorch/vision:v0.8.0', arch, pretrained=pretrained)
            # self.encoder = nn.Sequential(vgg.features, nn.Flatten(), nn.Linear(512, emb_dim))
            self.encoder = nn.Sequential(vgg.features,nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, emb_dim))
    def forward(self, x):
        emb = self.encoder(x)
        return F.normalize(emb, p=2, dim=-1)
