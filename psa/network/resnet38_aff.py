import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

import network.resnet38d
from tool import pyutils
import os.path as op
import os
import numpy as np


class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = torch.nn.Conv2d(4096, 256, 1, bias=False)

        self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)

        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_featuresize = int(448 // 8)
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(
            radius=5,
            size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from)
        self.ind_to = torch.from_numpy(self.ind_to)

        return

    def forward(self, x, to_dense=False, name=None, save_feature=False):

        d = super().forward_as_dict(x)

        f8_3 = F.elu(self.f8_3(d['conv4']))
        f8_4 = F.elu(self.f8_4(d['conv5']))
        f8_5 = F.elu(self.f8_5(d['conv6']))

        x = F.elu(self.f9(torch.cat([f8_3, f8_4, f8_5],
                                    dim=1)))  # [1,448,46,63]
        """ ============ save pixel feature ============ """
        if save_feature:
            if not op.exists("AFF_FEATURE_res38"):
                os.mkdir("AFF_FEATURE_res38")
            np.save(op.join("AFF_FEATURE_res38", name),
                    x.clone().cpu().numpy())

        if x.size(2) == self.predefined_featuresize and x.size(
                3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind_from, ind_to = pyutils.get_indices_of_pairs(
                5, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from)
            ind_to = torch.from_numpy(ind_to)

        # """ ============ save aff_feature ============ """
        # print("x.shape ", x.shape)
        # np.save(file=op.join("AFF_FEATURE", name),
        #         arr=x.clone().cpu().numpy(),
        #         allow_pickle=True)
        x = x.view(x.size(0), x.size(1), -1)  # [1,448,46*63=2898]

        ff = torch.index_select(
            x, dim=2, index=ind_from.cuda(non_blocking=True))  # [1,448,46*63]
        ft = torch.index_select(
            x, dim=2, index=ind_to.cuda(non_blocking=True))  # [1,448,46*63]
        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))  # [1,448,34,2310]
        aff = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))  # [1,34,2310]

        if to_dense:  # True
            aff = aff.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(
                ft.size(2), -1).contiguous().view(-1)  # [78540]
            indices = torch.stack([ind_from_exp, ind_to])  # [2,78540]
            indices_tp = torch.stack([ind_to, ind_from_exp])  # [2,78540]

            area = x.size(2)  # 2898

            # [2,2898]
            indices_id = torch.stack(
                [torch.arange(0, area).long(),
                 torch.arange(0, area).long()])

            # === sparse.floatTensor(coordinate,value)
            # e.g. 0 0 3
            #      3 0 5
            # coordinate =[[0 1 1],
            #    [2 0 2]]
            # value =[3,4,5]
            aff_mat = sparse.FloatTensor(
                torch.cat([indices, indices_id, indices_tp],
                          dim=1),  # coordinate
                torch.cat([aff, torch.ones([area]),
                           aff])).to_dense().cuda()  # scalar
            return aff_mat  # [2898,2898] tensor

        else:
            return aff

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d)
                    or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
