import os
import numpy as np

import torch
import torch.nn as nn

from .pu_net import PUNet
from ..drop_points import SORDefense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


class DUPNet(nn.Module):

    def __init__(self, sor_k=2, sor_alpha=1.1,
                 npoint=1024, up_ratio=4):
        super(DUPNet, self).__init__()

        self.npoint = npoint
        self.sor = SORDefense(k=sor_k, alpha=sor_alpha)
        self.pu_net = PUNet(npoint=self.npoint, up_ratio=up_ratio,
                            use_normal=False, use_bn=False, use_res=False)
        self.pu_net.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'pu-in_1024-up_4.pth')))
        self.pu_net = self.pu_net.cuda()
        self.pu_net.eval()

    def forward(self, x):
        with torch.enable_grad():
            x = self.sor(x)  # a list of pc
            x = x.transpose(1, 2)
            x = self.pu_net(x)  # [B, N * r, 3]
            x = x.transpose(1, 2)
        return x



