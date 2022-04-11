import os
from pickle import FALSE
import sys
import numpy as np
from collections import Iterable
import importlib
import open3d as o3d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from baselines import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))



class PointCloudAttack(object):
    def __init__(self, args):
        """Shape-invariant Adversarial Attack for 3D Point Clouds.
        """
        self.args = args
        self.device = args.device

        self.eps = args.eps
        self.normal = args.normal
        self.step_size = args.step_size
        self.num_class = args.num_class
        self.max_steps = args.max_steps
        self.top5_attack = args.top5_attack

        assert args.transfer_attack_method is None or args.query_attack_method is None
        assert not args.transfer_attack_method is None or not args.query_attack_method is None
        self.attack_method = args.transfer_attack_method if args.query_attack_method is None else args.query_attack_method

        self.build_models()
        self.defense_method = args.defense_method
        if not args.defense_method is None:
            self.pre_head = self.get_defense_head(args.defense_method)


    def build_models(self):
        """Build white-box surrogate model and black-box target model.
        """
        # load white-box surrogate models
        MODEL = importlib.import_module(self.args.surrogate_model)
        wb_classifier = MODEL.get_model(
            self.num_class,
            normal_channel=self.normal
        )
        wb_classifier = wb_classifier.to(self.device)
        # load black-box target models
        MODEL = importlib.import_module(self.args.target_model)
        classifier = MODEL.get_model(
            self.num_class,
            normal_channel=self.normal
        )
        classifier = classifier.to(self.args.device)
        # load model weights
        wb_classifier = self.load_models(wb_classifier, self.args.surrogate_model)
        classifier = self.load_models(classifier, self.args.target_model)
        # set eval
        self.wb_classifier = wb_classifier.eval()
        self.classifier = classifier.eval()


    def load_models(self, classifier, model_name):
        """Load white-box surrogate model and black-box target model.
        """
        model_path = os.path.join('./checkpoint/' + self.args.dataset, model_name)
        if os.path.exists(model_path + '.pth'):
            checkpoint = torch.load(model_path + '.pth')

        elif os.path.exists(model_path + '.t7'):
            checkpoint = torch.load(model_path + '.t7')

        elif os.path.exists(model_path + '.tar'):
            checkpoint = torch.load(model_path + '.tar')

        else:
            raise NotImplementedError

        try:
            if 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])

            elif 'model_state' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state'])

            else:
                classifier.load_state_dict(checkpoint)

        except:
            classifier = nn.DataParallel(classifier)
            classifier.load_state_dict(checkpoint)

        return classifier


    def CWLoss(self, logits, target, kappa=0, tar=False, num_classes=40):
        """Carlini & Wagner attack loss. 

        Args:
            logits (torch.cuda.FloatTensor): the predicted logits, [1, num_classes].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
        target_one_hot = Variable(torch.eye(num_classes).type(torch.cuda.FloatTensor)[target.long()].cuda())

        real = torch.sum(target_one_hot*logits, 1)
        if not self.top5_attack:
            ### top-1 attack
            other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
        else:
            ### top-5 attack
            other = torch.topk((1-target_one_hot)*logits - (target_one_hot*10000), 5)[0][:, 4]
        kappa = torch.zeros_like(other).fill_(kappa)

        if tar:
            return torch.sum(torch.max(other-real, kappa))
        else :
            return torch.sum(torch.max(real-other, kappa))


    def run(self, points, target):
        """Main attack method.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        if self.attack_method == 'ifgm_ours':
            return self.shape_invariant_ifgm(points, target)

        elif self.attack_method == 'simba':
            return self.simba_attack(points, target)

        elif self.attack_method == 'simbapp':
            return self.simbapp_attack(points, target)

        elif self.attack_method == 'ours':
            return self.shape_invariant_query_attack(points, target)

        else:
            NotImplementedError


    def get_defense_head(self, method):
        """Set the pre-processing based defense module.

        Args:
            method (str): defense method name.
        """
        if method == 'sor':
            pre_head = SORDefense(k=2, alpha=1.1)
        elif method == 'srs':
            pre_head = SRSDefense(drop_num=500)
        elif method == 'dupnet':
            pre_head = DUPNet(sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4)
        else:
            raise NotImplementedError
        return pre_head


    def get_normal_vector(self, points):
        """Calculate the normal vector.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).detach().cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normal_vec = torch.FloatTensor(pcd.normals).cuda().unsqueeze(0)
        return normal_vec


    def get_spin_axis_matrix(self, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        _, N, _ = normal_vec.shape
        x = normal_vec[:,:,0] # [1, N]
        y = normal_vec[:,:,1] # [1, N]
        z = normal_vec[:,:,2] # [1, N]
        assert abs(normal_vec).max() <= 1
        u = torch.zeros(1, N, 3, 3).cuda()
        denominator = torch.sqrt(1-z**2) # \sqrt{1-z^2}, [1, N]
        u[:,:,0,0] = y / denominator
        u[:,:,0,1] = - x / denominator
        u[:,:,0,2] = 0.
        u[:,:,1,0] = x * z / denominator
        u[:,:,1,1] = y * z / denominator
        u[:,:,1,2] = - denominator
        u[:,:,2] = normal_vec
        # revision for |z| = 1, boundary case.
        pos = torch.where(abs(z ** 2 - 1) < 1e-4)[1]
        u[:,pos,0,0] = 1 / np.sqrt(2)
        u[:,pos,0,1] = - 1 / np.sqrt(2)
        u[:,pos,0,2] = 0.
        u[:,pos,1,0] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,1] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,2] = 0.
        u[:,pos,2,0] = 0.
        u[:,pos,2,1] = 0.
        u[:,pos,2,2] = z[:,pos]
        return u.data


    def get_transformed_point_cloud(self, points, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        intercept = torch.mul(points, normal_vec).sum(-1, keepdim=True) # P \cdot N, [1, N, 1]
        spin_axis_matrix = self.get_spin_axis_matrix(normal_vec) # U, [1, N, 3, 3]
        translation_matrix = torch.mul(intercept, normal_vec).data # (P \cdot N) N, [1, N, 3]
        new_points = points + translation_matrix #  P + (P \cdot N) N, [1, N, 3]
        new_points = new_points.unsqueeze(-1) # P + (P \cdot N) N, [1, N, 3, 1]
        new_points = torch.matmul(spin_axis_matrix, new_points) # P' = U (P + (P \cdot N) N), [1, N, 3, 1]
        new_points = new_points.squeeze(-1).data # P', [1, N, 3]
        return new_points, spin_axis_matrix, translation_matrix


    def get_original_point_cloud(self, new_points, spin_axis_matrix, translation_matrix):
        """Calculate the spin-axis matrix.

        Args:
            new_points (torch.cuda.FloatTensor): the transformed point cloud with N points, [1, N, 3].
            spin_axis_matrix (torch.cuda.FloatTensor): the rotate matrix for transformation, [1, N, 3, 3].
            translation_matrix (torch.cuda.FloatTensor): the offset matrix for transformation, [1, N, 3, 3].
        """
        inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), new_points.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
        inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
        inputs = inputs.squeeze(-1) # P, [1, N, 3]
        return inputs


    def shape_invariant_ifgm(self, points, target):
        """Black-box I-FGSM based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)# * np.sqrt(3*1024))

        for i in range(self.max_steps):
            # P -> P', detach()
            new_points, spin_axis_matrix, translation_matrix = self.get_transformed_point_cloud(points, normal_vec)
            new_points = new_points.detach()
            new_points.requires_grad = True
            # P' -> P
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
            points = points.transpose(1, 2) # P, [1, 3, N]
            # get white-box gradients
            if not self.defense_method is None:
                logits = self.wb_classifier(self.pre_head(points))
            else:
                logits = self.wb_classifier(points)
            loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
            self.wb_classifier.zero_grad()
            loss.backward()
            # print(loss.item(), logits.max(1)[1], target)
            grad = new_points.grad.data # g, [1, N, 3]
            grad[:,:,2] = 0.
            # update P', P and N
            # # Linf
            # new_points = new_points - self.step_size * torch.sign(grad)
            # L2
            norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
            new_points = new_points - self.step_size * np.sqrt(3*1024) * grad / (norm[:, None, None] + 1e-9)
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix) # P, [1, N, 3]
            points = clip_func(points, ori_points)
            # points = torch.min(torch.max(points, ori_points - self.eps), ori_points + self.eps) # P, [1, N, 3]
            normal_vec = self.get_normal_vector(points) # N, [1, N, 3]

        with torch.no_grad():
            adv_points = points.data
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.transpose(1, 2).detach()))
            else:
                adv_logits = self.classifier(points.transpose(1, 2).detach())
            adv_target = adv_logits.data.max(1)[1]
        # print(target)
        # print(adv_target)
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        return adv_points, adv_target, (adv_logits.data.max(1)[1] != target).sum().item()


    def simba_attack(self, points, target):
        """Blaxk-box query-based SimBA attack.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        points = points[:,:,:3].data # P, [1, N, 3]
        # initialization
        query_costs = 0
        with torch.no_grad():
            points = points.transpose(1, 2)
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.detach()))
            else:
                adv_logits = self.classifier(points)
            adv_target = adv_logits.max(1)[1]
            query_costs += 1
        # if categorized wrong
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        if adv_target != target:
            return points.transpose(1, 2), adv_target, query_costs

        # constructing random list
        basis_list = []
        for j in range(points.shape[2]):
            for i in range(3):
                basis_list.append((i, j))
        basis_list = np.array(basis_list)
        np.random.shuffle(basis_list)

        # query loop
        i = 0
        best_loss = -999.
        while best_loss < 0 and i < len(basis_list):
            channel, idx = basis_list[i]
            for eps in {self.step_size, -self.step_size}:
                pert = torch.zeros_like(points).cuda() # \delta, [1, 3, N]
                pert[:,channel,idx] += eps
                inputs = points + pert
                with torch.no_grad():
                    if not self.defense_method is None:
                        logits = self.classifier(self.pre_head(inputs.detach()))
                    else:
                        logits = self.classifier(inputs.detach()) # [1, num_class]
                    query_costs += 1
                loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
                if loss.item() > best_loss:
                    # print(loss.item())
                    best_loss = loss.item()
                    points = points + pert
                    adv_target = logits.max(1)[1]
                    break
            i += 1
        # print(query_costs)
        # print(target)
        # print(adv_target)
        adv_points = points.transpose(1, 2).data
        if self.top5_attack:
            target_top_5 = logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        del grad

        return adv_points, adv_target, query_costs


    def simbapp_attack(self, points, target):
        """Blaxk-box query-based SimBA++ attack.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        points = points[:,:,:3].data # P, [1, N, 3]
        # initialization
        query_costs = 0
        with torch.no_grad():
            points = points.transpose(1, 2)
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.detach()))
            else:
                adv_logits = self.classifier(points)
            adv_target = adv_logits.max(1)[1]
            query_costs += 1
        # if categorized wrong
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        if adv_target != target:
            return points.transpose(1, 2), adv_target, query_costs

        # get white-box gradients
        points = points.detach()
        points.requires_grad = True
        logits = self.wb_classifier(points)
        loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
        self.wb_classifier.zero_grad()
        loss.backward()
        grad = points.grad.data # g, [1, 3, N]
        grad = abs(grad).reshape(-1)

        # # rank 
        # basis_list = []
        # for j in range(points.shape[2]):
        #     for i in range(3):
        #         basis_list.append((i, j, grad[0][i][j]))
        # sorted_basis_list = sorted(basis_list, key=lambda c: c[2], reverse=True)

        # query loop
        i = 0
        best_loss = -999.
        while best_loss < 0 and i < grad.shape[0]:
            # channel, idx, _ = sorted_basis_list[i]
            m = Categorical(grad)
            choice = m.sample()
            channel = int(choice % 3)
            idx = int(choice // 3)
            for eps in {self.step_size, -self.step_size}:
                pert = torch.zeros_like(points).cuda() # \delta, [1, 3, N]
                pert[:,channel,idx] += (eps + 0.1*torch.randn(1).cuda())
                inputs = points + pert
                with torch.no_grad():
                    if not self.defense_method is None:
                        logits = self.classifier(self.pre_head(inputs.detach()))
                    else:
                        logits = self.classifier(inputs.detach()) # [1, num_class]
                    query_costs += 1
                loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
                if loss.item() > best_loss:
                    # print(loss.item())
                    best_loss = loss.item()
                    points = points + pert
                    adv_target = logits.max(1)[1]
                    break
            i += 1
        # print(query_costs)
        # print(target)
        # print(adv_target)
        adv_points = points.transpose(1, 2).data
        if self.top5_attack:
            target_top_5 = logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        del grad, m

        return adv_points, adv_target, query_costs


    def shape_invariant_query_attack(self, points, target):
        """Blaxk-box query-based attack based on point-cloud sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        # initialization
        query_costs = 0
        with torch.no_grad():
            points = points.transpose(1, 2)
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.detach()))
            else:
                adv_logits = self.classifier(points)
            adv_target = adv_logits.max(1)[1]
            query_costs += 1
        # if categorized wrong
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        if adv_target != target:
            return points.transpose(1, 2), adv_target, query_costs

        # P -> P', detach()
        points = points.transpose(1, 2)
        new_points, spin_axis_matrix, translation_matrix = self.get_transformed_point_cloud(points.detach(), normal_vec)
        new_points = new_points.detach()
        new_points.requires_grad = True

        # P' -> P
        inputs = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
        inputs = torch.min(torch.max(inputs, ori_points - self.eps), ori_points + self.eps)
        inputs = inputs.transpose(1, 2) # P, [1, 3, N]

        # get white-box gradients
        logits = self.wb_classifier(inputs)
        loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
        self.wb_classifier.zero_grad()
        loss.backward()

        grad = new_points.grad.data # g, [1, N, 3]
        grad[:,:,2] = 0.
        new_points.requires_grad = False
        rankings = torch.sqrt(grad[:,:,0] ** 2 + grad[:,:,1] ** 2) # \sqrt{g_{x'}^2+g_{y'}^2}, [1, N]
        directions = grad / (rankings.unsqueeze(-1)+1e-16) # (g_{x'}/r,g_{y'}/r,0), [1, N, 3]

        # rank the sensitivity map in the desending order
        point_list = []
        for i in range(points.size(1)):
            point_list.append((i, directions[:,i,:], rankings[:,i].item()))
        sorted_point_list = sorted(point_list, key=lambda c: c[2], reverse=True)

        # query loop
        i = 0
        best_loss = -999.
        while best_loss < 0 and i < len(sorted_point_list):
            idx, direction, _ = sorted_point_list[i]
            for eps in {self.step_size, -self.step_size}:
                pert = torch.zeros_like(new_points).cuda()
                pert[:,idx,:] += eps * direction
                inputs = new_points + pert
                inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), inputs.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
                inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
                inputs = inputs.squeeze(-1).transpose(1, 2) # P, [1, 3, N]
                # inputs = torch.clamp(inputs, -1, 1)
                with torch.no_grad():
                    if not self.defense_method is None:
                        logits = self.classifier(self.pre_head(inputs.detach()))
                    else:
                        logits = self.classifier(inputs.detach()) # [1, num_class]
                    query_costs += 1
                loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
                if loss.item() > best_loss:
                    # print(loss.item())
                    best_loss = loss.item()
                    new_points = new_points + pert
                    adv_target = logits.max(1)[1]
                    break
            i += 1
        # print(query_costs)
        # print(target)
        # print(adv_target)
        adv_points = inputs.transpose(1, 2).data
        if self.top5_attack:
            target_top_5 = logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        del grad

        return adv_points, adv_target, query_costs

