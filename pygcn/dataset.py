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
from utils import load_img_name_list
from scipy import sparse


class graph_voc(data.Dataset):
    def __init__(self,
                 root=args.path4Image,
                 graph_type="AFF",
                 pseudo_label_path=os.path.join(
                     "..", "RES38_CRF_HALA_PSEUDO_LABEL(2020)"),
                 transofrms=None,
                 train=True,
                 test=False,
                 skip_untill=-1):
        self.pseudo_label_path = pseudo_label_path
        self.label_list = load_img_name_list(args.path4train_images)
        self.seg_label_dict = dict()
        self.test = test
        self.graph_type = graph_type  # AFF|RW|GT
        self.train_file = load_img_name_list(args.path4train_images)
        self.skip_untill = skip_untill
        # self.exclude_photoes = ['2010_004960', '2007_000032']

    def load_data(self,
                  graph_type='AFF',
                  path=None,
                  img_name=None,
                  path4data=os.path.join("..", "..", "..", "work",
                                         getpass.getuser(), "pygcn", "data")):
        """
        return adj, features, labels, idx_train, idx_test, rgbxy, img_name
        adj: sparse matrix
        features:
        """
        print('==================================================')
        print('{}: Loading nodes...'.format(img_name))

        # === get graph(affinity matrix) =======================
        # print("load Graph from:       {}".format(args.path4AffGraph))
        ful_path = os.path.join(
            path,
            args.graph_pre[graph_type] + img_name + args.graph_ext[graph_type])
        if graph_type == 'AFF':  # use affinity matrix of affiniityNet
            graph = np.load(ful_path)
        else:  # use affinity matrix of PSA random walk or Ground True
            graph = pkl.load(open(ful_path, "rb"))

        # === build symmetric adjacency matrix from graph
        adj = sp.coo_matrix(graph, dtype=np.float32)  # sparse matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(
            adj.T > adj)  # symmetrilize
        adj = normalize(adj + sp.eye(adj.shape[0]))

        # === load features,train_idx,test_idx, ========================
        # if os.path.basename(args.path4data) == "data":
        #     names = ['x', 'y', 'allx', 'ally', 'rgbxy']
        # else:
        #     names = ['train_idx', 'test_idx', 'feature', 'labels', 'rgbxy']
        """
        x: train_idx. list 
        y: test_idx. list
        allx: features for train and test. [sparse matrix]
        ally: labels, pseudo for train_idx, ground truth for test_idx. [np.array]
        rgbxy: another feature for comparison. [np.array]
        """
        # objects = []
        # for i in range(len(names)):
        #     with open(
        #             "{}.{}.{}".format(os.path.join(path4data, "ind"), img_name,
        #                               names[i]), 'rb') as f:
        #         if sys.version_info > (3, 0):
        #             objects.append(pkl.load(f, encoding='latin1'))
        #         else:
        #             objects.append(pkl.load(f))

        # train_idx, test_idx, features, labels, rgbxy = tuple(
        #     objects)  # xxx_idx format is list
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
        # if os.path.basename(args.path4data) == "data":
        #     # === trasfer to non-ont-hot ===
        #     labels = np.argmax(labels, axis=1)  # [H*W]
        #     # === transfer [idx1,idx2,...] to [True,False,....] for train_idx
        #     bool_arr = np.zeros_like(labels).astype(np.bool)
        #     bool_arr[train_idx] = True
        #     train_idx = bool_arr  # [H*W]
        #     labels[~train_idx] = 255  # ignore region = 255

        labels = Image.open(
            os.path.join(args.path4partial_label, img_name + '.png'))
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
        # === you can delete next 4 lines~~~~
        # idx_fg = np.where((labels != 0) & (labels != 255))
        # idx_bg = np.where(labels == 0)
        # train_idx = (labels != 255)
        # idx_train_fg_t = torch.LongTensor(np.intersect1d(idx_fg, train_idx))
        # idx_train_bg_t = torch.LongTensor(np.intersect1d(idx_bg, train_idx))

        # === transfer to tensor
        labels = torch.LongTensor(labels)
        # idx_train_t = torch.tensor(train_idx)
        label_fg_t = torch.LongTensor(label_fg)
        label_bg_t = torch.LongTensor(label_bg)
        # features = sp.coo_matrix(allx).tolil()

        # === shape: [1,448,H,W]
        img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
        H_origin, W_origin, C = img.shape
        H = int(np.ceil(H_origin / 8))
        W = int(np.ceil(W_origin / 8))

        # === reload features
        f_aff = np.load(os.path.join(args.path4node_feat, img_name + ".npy"))
        aff_version = os.path.basename(args.path4node_feat).split("_")[-1]
        f_aff = np.squeeze(f_aff)  # [448,H,W] for vgg| [448,H*W] for res38
        if aff_version != "v3":  # vgg16
            f_aff = np.reshape(f_aff, (np.shape(f_aff)[0], H * W))  # [448,H*W]
        # === RGB feature
        img_dn = Image.fromarray(img).resize(
            (W, H), Image.LANCZOS)  # [W,H 3] note!! resize的輸入是(W,H)
        img_dn = np.asarray(img_dn)  # [H,W,3]
        feature_rgbxy = np.zeros(shape=(H, W, 5))
        feature_rgbxy[:, :, :3] = img_dn / 255.  # conpress between 0~1
        # get xy feature
        for i in range(H):
            for j in range(W):
                feature_rgbxy[i, j, 3] = float(i)
                feature_rgbxy[i, j, 4] = float(j)
        allx = np.transpose(f_aff, [1, 0])  # [H*W,448]
        features = sparse.csr_matrix(allx)
        feat = torch.FloatTensor(np.array(features.todense()))
        rgbxy = torch.FloatTensor(feature_rgbxy)
        adj = torch.FloatTensor(adj.toarray())
        return adj, feat, labels, rgbxy, img_name, label_fg_t, label_bg_t

    def __getitem__(self, index):
        """
        return adj, feat, labels, idx_train_t, rgbxy, img_name, label_fg_t, label_bg_t
        """
        img_name = self.train_file[index]
        if index > self.skip_untill:
            return self.load_data(graph_type=self.graph_type,
                                  path=args.path4AffGraph,
                                  img_name=img_name,
                                  path4data=args.path4data)
        else:
            return None, None, None, None, None, None, None, None, None

    def __len__(self):
        return len(self.train_file)

    def gen_pseudo_label(self, img_name):
        """
        Return pseudo label, train index, test index (np array),
        ===
        idx_train_fg,idx_train_bg
        ===
        size
        - pseudo label: (HxW,)
        - train index: (HxW,)
        - test index: (HxW,)
        """
        complete_label_name = os.path.join(self.pseudo_label_path,
                                           img_name + '.png')
        mask_predit = Image.open(complete_label_name)
        seg_label = np.asarray(mask_predit)
        down_H, down_W = int(np.ceil(seg_label.shape[0] / 8)), int(
            np.ceil(seg_label.shape[1] / 8))
        print("H: {}, W: {}".format(seg_label.shape[0], seg_label.shape[1]))
        print("down_H: {}, down_W: {}".format(down_H, down_W))
        seg_label_dn = F.interpolate(
            torch.Tensor(seg_label).unsqueeze(dim=0).unsqueeze(dim=0),
            size=(down_H, down_W),
            mode='nearest').squeeze().cpu().numpy()

        print("seg_label_dn.shape ", seg_label_dn.shape)
        seg_label_dn = seg_label_dn.reshape(-1)  # [H*W]

        print("seg_label.shape ", seg_label.shape)
        print("seg_label.shape(-1) ", seg_label.reshape(-1).shape)
        print("seg_label_dn.shape(-1) ", seg_label_dn.reshape(-1).shape)
        self.seg_label_dict[img_name] = seg_label_dn

        train_idx = seg_label_dn != 255
        test_idx = ~train_idx
        idx_train_fg = (seg_label_dn != 0) & train_idx
        idx_train_bg = (seg_label_dn == 0) & train_idx
        print("idx_train_fg.shape ", idx_train_fg.shape)
        print("idx_train_fg ", np.sum(idx_train_fg))
        return seg_label_dn, train_idx, test_idx, idx_train_fg, idx_train_bg


if __name__ == "__main__":
    # test the corecness of the dataset
    path4AFF = "AFF_MAT_normalize"  # "aff_map_normalize" | "AFF_MAT_normalize"
    args.parse(
        path4data="data_v8",
        # path4data = data_RES_UP_CRF_DN_TRAIN| data_RES_UP_CRF_DN | data_RES_UP_CRF_DN_VAL
        hid_unit=40,
        max_epoch=250,
        drop_rate=.3,
        path4AffGraph=os.path.join("..", "psa", path4AFF),
        path4train_images=args.path4train_aug_images,
        path4partial_label="../psa/RES38_PARTIAL_PSEUDO_LABEL_DN",
        path4node_feat="../psa/AFF_FEATURE_res38")
    dataset = graph_voc()
    # print(dataset.train_file)
    import time
    from utils import show_timing
    t_start = time.time()
    for i, item in enumerate(dataset):
        adj, features, labels, rgbxy, img_name, label_fg_t, label_bg_t = item
        print(img_name)
        print("shape(features):", np.shape(features))
        print("labels.shape {}  time: {}".format(
            labels.shape, show_timing(t_start, time.time())))
