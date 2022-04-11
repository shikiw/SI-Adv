"""Grad-enable version SOR defense proposed by ICCV'19 paper DUP-Net"""
import numpy as np
import torch
import torch.nn as nn


class SORDefense(nn.Module):
    """Statistical outlier removal as defense.
    """

    def __init__(self, k=2, alpha=1.1, npoint=1024):
        """SOR defense.

        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(SORDefense, self).__init__()

        self.k = k
        self.alpha = alpha
        self.npoint = npoint

    def outlier_removal(self, x):
        """Removes large kNN distance points.

        Args:
            x (torch.FloatTensor): batch input pc, [B, K, 3]

        Returns:
            torch.FloatTensor: pc after outlier removal, [B, N, 3]
        """
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]
        threshold = mean + self.alpha * std  # [B]
        bool_mask = (value <= threshold[:, None])  # [B, K]
        sel_pc = x[0][bool_mask[0]].unsqueeze(0)
        sel_pc = self.process_data(sel_pc)
        for i in range(1, B):
            proc_pc = x[i][bool_mask[i]].unsqueeze(0)
            proc_pc = self.process_data(proc_pc)
            sel_pc = torch.cat([sel_pc, proc_pc], dim=0)
        return sel_pc

    def process_data(self, pc, npoint=None):
        """Process point cloud data to be suitable for
            PU-Net input.
        We do two things:
            sample npoint or duplicate to npoint.

        Args:
            pc (torch.FloatTensor): list input, [(N_i, 3)] from SOR.
                Need to pad or trim to [B, self.npoint, 3].
        """
        if npoint is None:
            npoint = self.npoint
        proc_pc = pc.clone()
        num = npoint // pc.size(1)
        for _ in range(num-1):
            proc_pc = torch.cat([proc_pc, pc], dim=1)
        num = npoint - proc_pc.size(1)
        duplicated_pc = proc_pc[:, :num, :]
        proc_pc = torch.cat([proc_pc, duplicated_pc], dim=1)
        assert proc_pc.size(1) == npoint
        return proc_pc

    def forward(self, x):
        with torch.enable_grad():
            x = x.transpose(1, 2)
            x = self.outlier_removal(x)
            x = self.process_data(x)  # to batch input
            x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    net = SORDefense()
    input = torch.randn(64, 3, 1024)
    input = torch.Tensor(input).cuda()
    input.requires_grad = True
    with torch.enable_grad():
        output = net(input)
    loss = torch.sum(output)
    grad = torch.autograd.grad(loss, [input])[0] # [1, C, N]
    print(grad.data)