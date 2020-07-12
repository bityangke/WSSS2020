import torch
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils import data
from utils import normalize

import os
import sys
from PIL import Image
import scipy.sparse as sp
import numpy as np
from cv2 import imread
import pickle as pkl
import networkx as nx
import getpass

from config import opt as args
from utils import load_img_name_list, normalize
from scipy import sparse
import time


def normalize_t(mx):
    """Row-normalize sparse matrix in tensor"""
    rowsum = torch.sum(mx, dim=1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diagflat(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


def preprocess_adj(aff_mat, device):
    adjT = torch.t(aff_mat)
    adj = torch.stack([aff_mat, adjT])
    adj, _ = adj.max(dim=0)
    return normalize_t(adj + torch.eye(adj.shape[0]).to(device))


class graph_voc(data.Dataset):
    def __init__(self,
                 root=args.path4Image,
                 graph_type="AFF",
                 transofrms=None,
                 train=True,
                 test=False,
                 skip_untill=-1,
                 start_idx=0,
                 end_idx=None,
                 device=None):
        self.label_list = load_img_name_list(args.path4train_images)
        self.seg_label_dict = dict()
        self.test = test
        self.graph_type = graph_type  # AFF|RW|GT
        self.train_file = load_img_name_list(args.path4train_images)
        self.skip_untill = skip_untill
        self.start_idx = start_idx
        if end_idx is None:
            self.end_idx = len(self.label_list)
        else:
            self.end_idx = end_idx
        self.device = device
        print("self.device: ", self.device)
        # self.ignore_list = [
        #     f.split(".")[0] for f in os.listdir(
        #         "/home/u7577591/pygcn/data/GCN_prediction/label/2020_7_9_17h"
        #     )]
        self.ignore_list = []

    def load_data(self,
                  graph_type='AFF',
                  path=None,
                  img_name=None,
                  path4data=None):
        """
        return adj, features, labels, idx_train, idx_test, rgbxy, img_name
        adj: sparse matrix
        features:
        """
        print('==================================================')
        # print('{}: Loading nodes...'.format(img_name))
        # === load features,train_idx,test_idx, ========================
        """
        graph: affinity matrix, .npy
        adj: affinity matrix after row normalize, .npy
        here we use graph to test if row normalize is work or not!
        """
        # === get graph(affinity matrix) =======================
        print("load Graph from:       {}".format(args.path4AffGraph))
        ful_path = os.path.join(
            path,
            args.graph_pre[graph_type] + img_name + args.graph_ext[graph_type])

        if graph_type == 'AFF':  # use affinity matrix of affiniityNet
            graph = np.load(ful_path)
        else:  # use affinity matrix of PSA random walk or Ground True
            graph = pkl.load(open(ful_path, "rb"))
        # === build symmetric adjacency matrix from graph
        # adj = sp.coo_matrix(graph, dtype=np.float32)  # sparse matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(
        #     adj.T > adj)  # symmetrilize
        # adj = normalize(adj + sp.eye(adj.shape[0]))
        # adj = torch.FloatTensor(adj.todense())
        # === build symmetric adjacency matrix from graph
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # device = torch.device("cuda:" + str(self.GPU_id))
        adj = preprocess_adj(
            torch.FloatTensor(graph).to(self.device), self.device)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        load_adjacency_mat = False
        if load_adjacency_mat:
            if getpass.getuser() == "u7577591":
                adj = np.load("../../../work/" + getpass.getuser() +
                              "/irn/AFF_MAT_normalize_IRNet_adj/" + img_name +
                              ".npy")
            else:
                adj = np.load(
                    "../../../work/" + getpass.getuser() +
                    "/daa732df-bd2b-4ef5-ae94-73da0de250fb/irn/AFF_MAT_normalize_IRNet_adj/"
                    + img_name + ".npy")
            adj = torch.FloatTensor(adj)
        """ === compute foreground & background === """
        # because (np.int8) will turn 255 -> -1,
        """ ===           generate bg label and fg label           ===
                              label 轉回non-one-hot
                                  put in tensor
                np.where(condition): 输出满足条件 (即非0) 元素的坐标
        """
        """
                Note!!!
                data/.ally do not contain 255
                and it is one-hot you can't use np.argmax directly
        """
        # === make seg label shape as [H*W]
        # print("label path: {}".format(args.path4partial_label_label,
        #                               img_name + '.png'))

        labels = Image.open(
            os.path.join(args.path4partial_label_label, img_name + '.png'))
        labels = np.asarray(labels)
        labels = np.reshape(labels, (-1)).astype(np.int16)

        # === np.int8 turn 255 -> -1, which is not what i want
        labels = np.where(labels == -1, 255,
                          labels)  # just to compatible old data
        # === generate FG/BG label
        label_fg = labels.copy()
        label_fg[label_fg == 0] = 255

        label_bg = labels.copy()
        label_bg[label_bg != 0] = 255

        # === transfer to tensor
        labels = torch.LongTensor(labels)
        label_fg_t = torch.LongTensor(label_fg)
        label_bg_t = torch.LongTensor(label_bg)
        # features = sp.coo_matrix(allx).tolil()

        # === shape: [1,448,H,W]
        img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
        H_origin, W_origin, C = img.shape
        aff_version = args.path4AffGraph.split("_")[-1]
        # ============ for IRNet
        if aff_version == "IRNet":
            # print("aff_version: ", args.path4AffGraph.split("_")[-1])
            H = int(np.ceil(H_origin / 4))
            W = int(np.ceil(W_origin / 4))
        else:  # ============ for PSA
            H = int(np.ceil(H_origin / 8))
            W = int(np.ceil(W_origin / 8))

        # === reload aff features
        # print("node fea.:   ".format(
        # os.path.join(args.path4node_feat, img_name + ".npy")))
        f_aff = np.load(os.path.join(args.path4node_feat, img_name + ".npy"))
        # print("f_aff.shape: ".format(f_aff.shape))
        f_aff = np.squeeze(f_aff)  # [448,H,W] for vgg| [448,H,W] for res38
        # print("aff_version: ".format(aff_version))
        if aff_version != "IRNet":
            f_aff = np.reshape(f_aff, (np.shape(f_aff)[0], H * W))  # [448,H*W]
        # print(f_aff.shape)
        f_aff = np.reshape(f_aff, (np.shape(f_aff)[0], H * W))
        # print(f_aff.shape)
        allx = np.transpose(f_aff, [1, 0])  # [H*W,448]
        # print("f_aff.shape:", allx.shape)
        features = sparse.csr_matrix(allx)
        # print("features.shape:", features.shape)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # === RGB feature
        img_dn = Image.fromarray(img).resize(
            (W, H), Image.LANCZOS
        )  # [W_downsample,H_downsample 3] note!! resize的輸入是(W,H)
        img_dn = np.asarray(img_dn)  # [H_downsample,H_downsample,3]
        rgbxy = np.zeros(shape=(H, W, 5))
        rgbxy[:, :, :3] = img_dn / 255.  # conpress between 0~1

        # get xy feature
        for i in range(H):
            for j in range(W):
                rgbxy[i, j, 3] = float(i) / H
                rgbxy[i, j, 4] = float(j) / W

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        feat = torch.FloatTensor(np.array(features.todense()))
        rgbxy_t = torch.FloatTensor(rgbxy)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)

        return {
            "adj_t": adj,
            "features_t": feat,
            "labels_t": labels,
            "rgbxy_t": rgbxy_t,
            "img_name": img_name,
            "label_fg_t": label_fg_t,
            "label_bg_t": label_bg_t
        }

    def __getitem__(self, index):
        """
        return adj, feat, labels, idx_train_t, rgbxy, img_name, label_fg_t, label_bg_t
        """
        img_name = self.train_file[index]
        if self.start_idx <= index < self.end_idx:
            print("start_idx: {}  end_idx: {}".format(self.start_idx,
                                                      self.end_idx))
            if img_name in self.ignore_list:
                print("[{}] ignore: {}".format(index, img_name))
                return None
            return self.load_data(graph_type=self.graph_type,
                                  path=args.path4AffGraph,
                                  img_name=img_name,
                                  path4data=args.path4data)
        else:
            return None

    def __len__(self):
        return len(self.train_file)


if __name__ == "__main__":
    # test the corecness of the dataset
    args.parse(hid_unit=40,
               max_epoch=250,
               drop_rate=.3,
               path4train_images=args.path4train_aug_images,
               path4AffGraph=os.path.join("..", "psa", "AFF_MAT_normalize"),
               path4partial_label_label=os.path.join(
                   "..", "psa", "RES38_PARTIAL_PSEUDO_LABEL_DN"),
               path4node_feat=os.path.join("..", "psa", "AFF_FEATURE_res38"))
    if getpass.getuser() == "u7577591":
        args.path4node_feat = os.path.join("/work/u7577591/",
                                           "irn/AFF_FEATURE_res50_W")
        args.path4partial_label_label = "data/partial_pseudo_label/" + "label/" + "RES_CAM_TRAIN_AUG_PARTIAL_PSEUDO_LABEL" + "@PIL_near@confident_ratio_" + "0.3_cam_DN_johnney"
        args.path4AffGraph = os.path.join("/work/u7577591/irn",
                                          "AFF_MAT_normalize_IRNet")

    dataset = graph_voc()
    import time
    from utils import show_timing
    t_start = time.time()
    for i, item in enumerate(dataset, start=1):
        data = item
        # adj, features, labels, rgbxy, img_name, label_fg_t, label_bg_t = item
        t_now = time.time()
        print("==================================")
        print("[{}/{}] time: {}".format(i, len(dataset),
                                        show_timing(t_start, t_now)))
        t_start = t_now
        print("adj ", data["adj_t"].shape)
        print("features ", data["features_t"].shape)
        print("labels ", data["labels_t"].shape)
        print("rgbxy ", data["rgbxy_t"].shape)
        print("img_name ", data["img_name"])
        print("label_fg_t ", data["label_fg_t"].shape)
        print("label_bg_t ", data["label_bg_t"].shape)
