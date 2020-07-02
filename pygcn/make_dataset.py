import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F

from PIL import Image
from cv2 import imread
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy.misc
import random
import pickle
import matplotlib.cm as cm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
from utils import show_timing
import fire
from config import opt as args
from utils import load_img_name_list, colors_map, compute_seg_label, load_image_label_from_xml


def gen_label_from_data(destination,
                        root=os.path.join("..", "..", "..", "work", "User",
                                          "pygcn", "data")):
    """
    - Visualize the pseudo label used to train GCN , (IoU=54.63)    
    ===
    - the file (.ally and .x) in work/UserName/pygcn/data/  is need
    - assume that the training data is use pickle to save
    - e.g. xxx.allx | xxx.ally | xxx.x | xxx.y | ...
    - assume .ally is prediction ont-hot label, type: np.array, shape: [H*W,21] 
    - H=np.ceil(H_original_size/8), W also so the same way
    - assume .x is train_idx, type: list, shape: [<H*W], element: int not bool 
    """
    import pickle as pkl
    import sys
    t_start = time.time()
    file_name_list = load_img_name_list(args.path4train_images)
    print("file_name_list: ", len(file_name_list))
    for file_name in file_name_list:
        print("Read... {}".format(file_name))
        """
        x: train_idx. [list]
        y: test_idx. [list]
        allx: features for train and test. [sparse matrix]
        ally: labels, pseudo for train_idx, ground truth for test_idx. [np.array]
        rgbxy: another feature for comparison. [np.array]
        """
        seg_label = None
        extension_list = ["ally", "x"]
        objects = []
        for ext in extension_list:
            with open(
                    "{}.{}.{}".format(os.path.join(args.path4data, "ind"),
                                      file_name, ext), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        seg_label, train_idx = tuple(objects)
        print("seg_label.shape ", seg_label.shape)
        print("type(train_idx[0]): ", type(train_idx[0]))
        """
        find H,W
        =============================================
        """
        GT_seg = Image.open(os.path.join(args.path4Class, file_name + '.png'))
        print("np.array(GT_seg).shape:", np.array(GT_seg).shape)
        H_origin, W_origin = np.array(GT_seg).shape
        H, W = int(np.ceil(H_origin / 8)), int(np.ceil(W_origin / 8))
        seg_label_HW = np.argmax(seg_label, axis=1)
        """
        reshape the seg_label from [H*W,21] to [H,W]
        =============================================
        """

        seg_label_HW = seg_label_HW.reshape(H, W)
        train_idx_bool = np.zeros_like(seg_label_HW.reshape(-1))
        print("train_idx_bool.shape ", train_idx_bool.shape)
        train_idx_bool[:] = False
        train_idx_bool[train_idx] = True
        train_idx_bool = train_idx_bool.reshape(H, W)
        print("np.array(train_idx).shape: ", np.array(train_idx).shape)
        """
        upsampling label and savi it in a visualizable way
        =============================================
        """
        seg_label_HW_UP = F.interpolate(
            torch.FloatTensor(seg_label_HW[np.newaxis, np.newaxis]),
            size=(H_origin, W_origin),
            mode='nearest').squeeze().cpu().numpy()

        train_idx_bool_UP = np.zeros_like(seg_label_HW_UP.reshape(-1))
        train_idx_bool_UP = F.interpolate(
            torch.FloatTensor(train_idx_bool[np.newaxis, np.newaxis]),
            size=(H_origin, W_origin),
            mode='nearest').squeeze().cpu().numpy()
        train_idx_bool_UP = np.asarray(train_idx_bool_UP, dtype=np.bool)
        """
        save it!
        =============================================
        """
        folder_list = [
            "VGG_CAM_PSEUDO_LABEL(54p63)HW", "VGG_CAM_PSEUDO_LABEL(54p63)HW_UP"
        ]
        for folder in folder_list:
            if not os.path.exists(folder):
                os.mkdir(folder)
        # save upsampling  label
        print("save upsamping  label!!!")
        scipy.misc.toimage(np.where(train_idx_bool_UP, seg_label_HW_UP, 255),
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join("VGG_CAM_PSEUDO_LABEL(54p63)HW_UP",
                                            file_name + '.png'))
        # save dowsampling  label (used to train GCN)
        print("save original label!!!")
        scipy.misc.toimage(np.where(train_idx_bool, seg_label_HW, 255),
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join("VGG_CAM_PSEUDO_LABEL(54p63)HW",
                                            file_name + '.png'))


def gen_one_data(img_name,
                 predict_root=None,
                 destination="data(2020)",
                 destination4visulization="RES38_PSEUDO_LABEL"):
    """
    Generate Data for Training
    ===
    - save data in `./data`,include:
    - rgbxy: the simplest features
    - allx: the affinity features, each feature of node is 400+ dimensions
    - x: train_idx
    - y: test_idx
    - ally: labels for all nodes, train label:pseudo, test label: ground truth  
    
    名詞解釋
    ---
    - high_crf: 對於background比較confident的label
    - low_crf: 對於object比較confident的label
    - label: 結合以上兩者得的pseudo labels (confident pixel label)
    - mask: 0,1矩陣, 記錄所有confident pixel的位置
    """
    if predict_root is None:
        predict_root = args.path4CAM
    if not os.path.exists(destination):
        os.mkdir(destination)
    if not os.path.exists(destination4visulization + '_DN_UP'):
        os.mkdir(destination4visulization + '_DN_UP')

    img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
    H_origin, W_origin, C = img.shape
    print("img.shape", img.shape)
    H = int(np.ceil(H_origin / 8))
    W = int(np.ceil(W_origin / 8))
    """=== generate pseudo label and idx for train,test"""
    cams = np.zeros((args.num_class - 1, H_origin, W_origin))  # [20,H,W]
    dict_np = np.load(os.path.join(predict_root, img_name + '.npy')).item()

    cam_label = np.zeros(20)
    """ === downsampling label_score for sava in .npy === """
    # img_dn = Image.fromarray(img).resize(
    #     (W, H), Image.LANCZOS)  # [W,H 3] note!! resize的輸入是(W,H)
    # img_dn = np.asarray(img_dn)  # [H,W,3]
    for key, cam in dict_np.items():
        # downsampling
        # cam_dn = F.interpolate(torch.tensor(cam[np.newaxis, np.newaxis, :, :]),
        #                        size=(H, W),
        #                        mode="bilinear",
        #                        align_corners=False).squeeze().numpy()
        # cams_dn[key] = cam_dn  # note! cam label from 0-20 not include BG
        cam_label[key] = 1
        cams[key] = cam
    """ === CRF interset CAMla+CAMha. sava label_score in .npy === """
    cam_label = cam_label.astype(np.uint8)
    seg_label, seg_score = compute_seg_label(ori_img=img,
                                             cam_label=cam_label,
                                             norm_cam=cams,
                                             use_crf=True)
    """ === downsampling label_score for sava in .npy === """
    # === downsampling score may used by random walk
    img_dn = Image.fromarray(img).resize(
        (W, H), Image.LANCZOS)  # [W,H 3] note!! resize的輸入是(W,H)
    img_dn = np.asarray(img_dn)  # [H,W,3]
    seg_score_dn = F.interpolate(torch.tensor(seg_score[np.newaxis, :, :, :]),
                                 size=(H, W),
                                 mode="bilinear",
                                 align_corners=False).squeeze().numpy()
    seg_label_dn = F.interpolate(torch.tensor(
        seg_label[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
                                 size=(H, W),
                                 mode="nearest").squeeze().numpy()
    seg_label_dn_up = F.interpolate(torch.tensor(
        seg_label_dn[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
                                    size=(H_origin, W_origin),
                                    mode="nearest").squeeze().numpy()

    # seg_label_dn = np.argmax(seg_score_dn, axis=0)
    """ === save seg_label_dn_up in destination4visulization + '_DN_UP' === """
    scipy.misc.toimage(seg_label_dn_up,
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode='P').save(
                           os.path.join(destination4visulization + '_DN_UP',
                                        "{}.png".format(img_name)))

    def save_pseudo_label(seg_score,
                          seg_label,
                          destination,
                          img_name="2007_000032",
                          save_npy=True):
        """
        Save Label and Label Score to `.png` and dictionary
        ===
        - label would be upsample to save
        - `img_name`: str, only file name, not include extension or path
        - `seg_score`: numpy array, shape: [num_class,H,W] 
        - `seg_label`: numpy array, shape: [H,W] 
        """
        if not os.path.exists(destination):
            os.mkdir(destination)
        destination_np = destination + "_SCORE"
        if not os.path.exists(destination_np):
            os.mkdir(destination_np)

        pseudo_label_dict = dict()
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        pseudo_label_dict[0] = seg_score[0]
        # key range from 0~20 if you use VOC dataset
        for key in img_label:  # img_label +1 = segmentation_label
            pseudo_label_dict[int(key)] = seg_score[int(key + 1)]
        # save score
        if save_npy:
            np.save(os.path.join(destination_np, img_name), pseudo_label_dict)
        # Save label mask
        scipy.misc.toimage(seg_label,
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(destination,
                                            "{}.png".format(img_name)))

    save_pseudo_label(seg_score=seg_score_dn,
                      seg_label=seg_label_dn,
                      img_name=img_name,
                      destination=destination4visulization + '_DN',
                      save_npy=True)

    # save_pseudo_label(seg_score=seg_score,
    #                   seg_label=seg_label,
    #                   img_name=img_name,
    #                   destination=destination4visulization,
    #                   save_npy=True)
    """ === upsampling label for save in .png === """
    seg_label_up = F.interpolate(torch.DoubleTensor(
        seg_label_dn[np.newaxis, np.newaxis, :, :]),
                                 size=(H_origin, W_origin),
                                 mode="nearest").squeeze().numpy()
    save_pseudo_label(seg_score=seg_score,
                      seg_label=seg_label,
                      img_name=img_name,
                      destination=destination4visulization + '_UP',
                      save_npy=True)
    """=== get faetures 1: affinity map ==="""
    # === shape: [1,448,H,W]
    f_aff = np.load(os.path.join(args.path4PsaFeature, img_name + ".npy"))
    print("H {}, W {} ,H * W {}".format(H, W, H * W))
    aff_version = os.path.basename(args.path4PsaFeature).split("_")[-1]
    f_aff = np.squeeze(f_aff)  # [448,H,W] for vgg16| [448,H*W] for res38
    print("aff_version ", aff_version)

    # if aff_version != "v3":  # vgg16
    #     f_aff = np.reshape(f_aff, (np.shape(f_aff)[0], H * W))  # [448,H*W]
    allx = np.transpose(f_aff, [1, 0])  # [H*W,448]
    print("args.path4PsaFeature ", args.path4PsaFeature)
    print("aff_feature shape ", np.shape(f_aff))
    """=== get features 2: rgbxy ==="""
    feature_rgbxy = np.zeros(shape=(H, W, 5))
    # get rgb features
    feature_rgbxy[:, :, :3] = img_dn / 255.  # conpress between 0~1
    # get xy feature
    for i in range(H):
        for j in range(W):
            feature_rgbxy[i, j, 3] = float(i)
            feature_rgbxy[i, j, 4] = float(j)
    """=== make pixel index for train & test ==="""
    print("=== make pixel index for train & test ===")
    label = np.reshape(seg_label_dn, (-1)).astype(np.int16)
    # 括号( )既可以表示tuple，又可以表示数学公式中的小括号。
    # 所以如果元组只有1个元素，就必须加一个逗号。
    x = (label != 255)  # train_idx
    y = ~x  # test_idx
    # to one-hot labels
    # label_one_hot = np.zeros(shape=(label.size, args.num_class),
    #                          dtype=np.uint8)
    # label_one_hot = np.eye(args.num_class)[label]
    # identity = np.eye(args.num_class)
    # for idx, label_ in enumerate(label):
    # index need to be int
    # label_one_hot[idx] = identity[label]

    # to csr_matrix

    rgbxy = np.reshape(feature_rgbxy, (H * W, np.shape(feature_rgbxy)[2]))
    """rgbxy = sparse.csr_matrix(rgbxy)"""
    feature = sparse.csr_matrix(allx)
    """print("allx", feature.shape)"""

    print("save data...")
    # save features(rgbxy) [train+test]
    pickle.dump(
        rgbxy,
        open(os.path.join(destination, "ind." + img_name + ".rgbxy"), "wb"))
    # save features(affinity feature) [train+test]
    pickle.dump(
        feature,
        open(os.path.join(destination, "ind." + img_name + ".feature"), "wb"))
    # save train_idx
    pickle.dump(
        x,
        open(os.path.join(destination, "ind." + img_name + ".train_idx"),
             "wb"))
    # save test_idx
    pickle.dump(
        y,
        open(os.path.join(destination, "ind." + img_name + ".test_idx"), "wb"))
    # save all labels [train:pseudo ; test: ground trueth]
    pickle.dump(
        label,
        open(os.path.join(destination, "ind." + img_name + ".labels"), "wb"))

    # =====================================


def gen_dataset(predict_root=None,
                destination="data_RES(2020)",
                destination4visulization="RES38_PSEUDO_LABEL",
                img_list_path=None):
    """
        Generate all data for a DataSet
        ===
        - save data in `./data`,include:
        - rgbxy: the simplest features, nparray
        - feature: the affinity features, each feature of node is 400+ dimensions, nparray
        - train_idx: nparray, value: bool, shape: [H*W]
        - test_idx: nparray, value: bool, shape: [H*W]
        - label: one-hot label nparray, value: int, shape: [H*W]
        """
    t_start = time.time()
    print("")
    if predict_root is None:
        predict_root = os.path.join("..", "psa", "RES_CAM__")
    # img_list = read_file('./train.txt')
    img_list = load_img_name_list(img_list_path)
    len_img_list = len(img_list)
    print("len_img_list ", len_img_list)
    # exclude = [
    #     "2007_000032", "2008_002350", "2008_004633", "2008_008057",
    #     "2010_000197"
    # ]
    for idx, name in enumerate(img_list, start=1):  # segmentation_class:
        print("===========\n[{}/{}]: {} generate data...".format(
            idx, len_img_list, name))
        gen_one_data(img_name=name,
                     predict_root=predict_root,
                     destination=destination,
                     destination4visulization=destination4visulization)
    show_timing(time_start=t_start, time_end=time.time())


def gen_partial_label(img_name,
                      predict_root=None,
                      destination="data(2020)",
                      destination4visulization="RES38_PSEUDO_LABEL"):
    """
    """
    if predict_root is None:
        predict_root = args.path4CAM
    if not os.path.exists(destination):
        os.mkdir(destination)
    if not os.path.exists(destination4visulization + '_DN_UP'):
        os.mkdir(destination4visulization + '_DN_UP')

    img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
    H_origin, W_origin, C = img.shape
    print("img.shape", img.shape)
    H = int(np.ceil(H_origin / 8))
    W = int(np.ceil(W_origin / 8))
    """=== generate pseudo label and idx for train,test ==="""
    cams = np.zeros((args.num_class - 1, H_origin, W_origin))  # [20,H,W]
    dict_np = np.load(os.path.join(predict_root, img_name + '.npy'),
                      allow_pickle=True).item()
    cam_label = np.zeros(20)
    # =================================
    # save CAM in dictionary
    # i think this part can be delete,
    # just set cams = np.load(.......)
    # =================================
    for key, cam in dict_np.items():
        cam_label[key] = 1
        cams[key] = cam
    """ === CRF interset CAMla+CAMha. sava label_score in .npy === """
    cam_label = cam_label.astype(np.uint8)
    seg_label, seg_score = compute_seg_label(ori_img=img,
                                             cam_label=cam_label,
                                             norm_cam=cams,
                                             use_crf=True)
    """ === downsampling label_score for sava in .npy === """
    # === downsampling score may used by random walk
    img_dn = Image.fromarray(img).resize(
        (W, H), Image.LANCZOS)  # [W,H 3] note!! resize的輸入是(W,H)
    img_dn = np.asarray(img_dn)  # [H,W,3]
    seg_score_dn = F.interpolate(torch.tensor(seg_score[np.newaxis, :, :, :]),
                                 size=(H, W),
                                 mode="bilinear",
                                 align_corners=False).squeeze().numpy()
    # =======================================================================
    # FYI,By deepLabv2 , it say that
    # Image.resize can do better downsample than cv2 and torch interpolation
    # =======================================================================
    seg_label_dn = F.interpolate(torch.tensor(
        seg_label[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
                                 size=(H, W),
                                 mode="nearest").squeeze().numpy()
    # === just for visualization
    seg_label_dn_up = F.interpolate(torch.tensor(
        seg_label_dn[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
                                    size=(H_origin, W_origin),
                                    mode="nearest").squeeze().numpy()

    # seg_label_dn = np.argmax(seg_score_dn, axis=0)
    """ === save seg_label_dn_up in destination4visulization + '_DN_UP' === """
    scipy.misc.toimage(seg_label_dn_up,
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode='P').save(
                           os.path.join(destination4visulization + '_DN_UP',
                                        "{}.png".format(img_name)))

    def save_pseudo_label(seg_score,
                          seg_label,
                          destination,
                          img_name="2007_000032",
                          save_npy=True):
        """
        Save Label and Label Score to `.png` and dictionary
        ===
        - label would be upsample to save
        - `img_name`: str, only file name, not include extension or path
        - `seg_score`: numpy array, shape: [num_class,H,W] 
        - `seg_label`: numpy array, shape: [H,W] 
        """
        if not os.path.exists(destination):
            os.mkdir(destination)
        destination_np = destination + "_SCORE"
        if not os.path.exists(destination_np):
            os.mkdir(destination_np)

        pseudo_label_dict = dict()
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        pseudo_label_dict[0] = seg_score[0]
        # key range from 0~20 if you use VOC dataset
        for key in img_label:  # img_label +1 = segmentation_label
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # pseudo_label_dict[int(key)] or pseudo_label_dict[int(key+1)] ???
            pseudo_label_dict[int(key + 1)] = seg_score[int(key + 1)]
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # save score
        if save_npy:
            np.save(os.path.join(destination_np, img_name), pseudo_label_dict)
        # Save label mask
        scipy.misc.toimage(seg_label,
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(destination,
                                            "{}.png".format(img_name)))

    save_pseudo_label(seg_score=seg_score_dn,
                      seg_label=seg_label_dn,
                      img_name=img_name,
                      destination=destination4visulization + '_DN',
                      save_npy=True)
    """ === upsampling label for save in .png === """
    save_pseudo_label(seg_score=seg_score,
                      seg_label=seg_label,
                      img_name=img_name,
                      destination=destination4visulization + '_UP',
                      save_npy=True)
    """=== make pixel index for train & test ==="""
    print("=== make pixel index for train & test ===")
    label = np.reshape(seg_label_dn, (-1)).astype(np.int16)
    # 括号( )既可以表示tuple，又可以表示数学公式中的小括号。
    # 所以如果元组只有1个元素，就必须加一个逗号。
    x = (label != 255)  # train_idx


def gen_partial_label_with_ratio(img_name,
                                 predict_root=None,
                                 cam_mode="cam",
                                 destination4visulization="RES38_PSEUDO_LABEL",
                                 destination4logit=None,
                                 confident_region=1.,
                                 show_infromation=False):
    """
    Generate partial pseudo label with ratio, ignore (1-ratio)
    ---
    the data folder will be generated:
    - destination4visulization + '_DN_UP'
    - destination4visulization + '_DN'
    - destination4visulization + '_UP'
    - destination4visulization + "_DN_SCORE"
    - destination4visulization + "_UP_SCORE"
    """
    if not os.path.exists(destination4visulization + '_DN_UP'):
        os.mkdir(destination4visulization + '_DN_UP')

    img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
    H_origin, W_origin, C = img.shape
    # print("img.shape", img.shape)
    H = int(np.ceil(H_origin / 4))
    W = int(np.ceil(W_origin / 4))
    """=== generate pseudo label and idx for train,test ==="""
    cams = np.zeros((args.num_class - 1, H_origin, W_origin))  # [20,H,W]
    dict_np = np.load(os.path.join(predict_root, img_name + '.npy'),
                      allow_pickle=True).item()
    cam_label = np.zeros(20)
    # =================================
    # save CAM in dictionary
    # i think this part can be delete,
    # just set cams = np.load(.......)
    # =================================
    for key, cam in dict_np.items():
        if cam_mode == "rw":
            if key == 0:
                continue
            key = key - 1
        cam_label[key] = 1
        cams[key] = cam
    """ === CRF interset CAMla+CAMha. sava label_score in .npy === """
    cam_label = cam_label.astype(np.uint8)
    # [C,H_up,W_up]
    seg_label, seg_score = compute_seg_label(ori_img=img,
                                             cam_label=cam_label,
                                             norm_cam=cams,
                                             use_crf=True,
                                             confident_region=confident_region)
    """ === downsampling label_score for sava in .npy === """
    # === downsampling score may used by random walk
    img_dn = Image.fromarray(img).resize(
        (W, H), Image.LANCZOS)  # [W,H 3] note!! resize的輸入是(W,H)
    img_dn = np.asarray(img_dn)  # [H,W,3]
    seg_score_dn = F.interpolate(torch.tensor(seg_score[np.newaxis, :, :, :]),
                                 size=(H, W),
                                 mode="bilinear",
                                 align_corners=False).squeeze().numpy()
    # [C,H,W]
    # seg_label_dn_ = F.interpolate(torch.tensor(seg_score[np.newaxis, :, :],
    #                                            dtype=torch.float64),
    #                               size=(H, W),
    #                               mode="bilinear",
    #                               align_corners=False).squeeze().numpy()
    # seg_label_dn = np.argmax(seg_label_dn_, axis=0)
    # include ignore label 255
    seg_label = seg_label.astype(np.uint8)
    # === using PIL to downsample
    seg_label_PIL_dn = Image.fromarray(seg_label).resize(
        (W, H), Image.NEAREST)  # [W,H 3] note!! resize的輸入是(W,H)
    seg_label_dn = np.asarray(seg_label_PIL_dn)  # [H,W]
    # === using torch.bilinear to downsample
    seg_label_torch_dn = seg_score_dn.argmax(axis=0)  # [H,W]
    # === vote
    seg_label_dn = np.where(seg_label_torch_dn == seg_label_PIL_dn,
                            seg_label_torch_dn, 255)
    if show_infromation:  # show infromation
        print("seg_label.shape ", seg_label.shape)
        print("np.unique(seg_label) ", np.unique(seg_label))
        print("np.unique(seg_label_dn) ", np.unique(seg_label_dn))
    # [H,W]

    # seg_score_dn = Image.fromarray(seg_score_chw).resize(
    #     (W, H), Image.LANCZOS)  # [H,W,3]
    # seg_label_dn = np.argmax(seg_score_dn, axis=2)
    # =======================================================================
    # FYI,By deepLabv2 , it say that
    # Image.resize can do better downsample than cv2 and torch interpolation
    # =======================================================================
    # ==== use torch to down sample
    # seg_label_dn = F.interpolate(torch.tensor(
    #     seg_label[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
    #                              size=(H, W),
    #                              mode="nearest").squeeze().numpy()
    # === just for visualization
    seg_label_dn_up = F.interpolate(torch.tensor(
        seg_label_dn[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
                                    size=(H_origin, W_origin),
                                    mode="nearest").squeeze().numpy()

    # seg_label_dn = np.argmax(seg_score_dn, axis=0)
    """ === save seg_label_dn_up in destination4visulization + '_DN_UP' === """
    scipy.misc.toimage(seg_label_dn_up,
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode='P').save(
                           os.path.join(destination4visulization + '_DN_UP',
                                        "{}.png".format(img_name)))

    def save_pseudo_label(seg_score,
                          seg_label,
                          destination,
                          img_name="2007_000032",
                          save_npy=True):
        """
        Save Label and Label Score to `.png` and dictionary
        ===
        - label would be upsample to save
        - `img_name`: str, only file name, not include extension or path
        - `seg_score`: numpy array, shape: [num_class,H,W] 
        - `seg_label`: numpy array, shape: [H,W] 
        """
        if not os.path.exists(destination):
            os.mkdir(destination)

        pseudo_label_dict = dict()
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        pseudo_label_dict[0] = seg_score[0]
        # key range from 0~20 if you use VOC dataset
        for key in img_label:  # img_label +1 = segmentation_label
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # pseudo_label_dict[int(key)] or pseudo_label_dict[int(key+1)] ???
            pseudo_label_dict[int(key + 1)] = seg_score[int(key + 1)]
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # save score
        if save_npy:
            destination_np = destination4logit
            if not os.path.exists(destination_np):
                os.mkdir(destination_np)
            np.save(os.path.join(destination_np, img_name), pseudo_label_dict)
        # Save label mask
        scipy.misc.toimage(seg_label,
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(destination,
                                            "{}.png".format(img_name)))

    """ === downsampling label for save in .png === """
    save_pseudo_label(seg_score=seg_score_dn,
                      seg_label=seg_label_dn,
                      img_name=img_name,
                      destination=destination4visulization + '_DN',
                      save_npy=False)
    """ === upsampling label for save in .png === """
    save_pseudo_label(seg_score=seg_score,
                      seg_label=seg_label,
                      img_name=img_name,
                      destination=destination4visulization + '_UP',
                      save_npy=False)


def gen_EX_PPL_ratio(img_name,
                     predict_root=None,
                     cam_mode="cam",
                     destination4visulization="RES38_PSEUDO_LABEL",
                     destination4logit=None,
                     confident_region=1.,
                     show_infromation=False):
    """
    Generate extend version of the partial pseudo label with ratio, ignore (1-ratio)
    ---
    the data folder will be generated:
    - destination4visulization + '_DN_UP'
    - destination4visulization + '_DN'
    - destination4visulization + '_UP'
    - destination4visulization + "_DN_SCORE"
    - destination4visulization + "_UP_SCORE"
    """
    if not os.path.exists(destination4visulization + '_DN_UP'):
        os.mkdir(destination4visulization + '_DN_UP')

    img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
    H_origin, W_origin, C = img.shape
    # print("img.shape", img.shape)
    H = int(np.ceil(H_origin / 8))
    W = int(np.ceil(W_origin / 8))
    """=== generate pseudo label and idx for train,test ==="""
    cams = np.zeros((args.num_class - 1, H_origin, W_origin))  # [20,H,W]
    dict_np = np.load(os.path.join(predict_root, img_name + '.npy'),
                      allow_pickle=True).item()
    cam_label = np.zeros(20)
    # =================================
    # save CAM in dictionary
    # i think this part can be delete,
    # just set cams = np.load(.......)
    # =================================
    for key, cam in dict_np.items():
        if cam_mode == "rw":
            if key == 0:
                continue
            key = key - 1
        cam_label[key] = 1
        cams[key] = cam
    """ === CRF interset CAMla+CAMha. sava label_score in .npy === """
    cam_label = cam_label.astype(np.uint8)
    # [C,H_up,W_up]
    seg_label, seg_score = compute_seg_label(ori_img=img,
                                             cam_label=cam_label,
                                             norm_cam=cams,
                                             use_crf=True,
                                             confident_region=confident_region)
    """ === downsampling label_score for sava in .npy === """
    # === downsampling score may used by random walk
    img_dn = Image.fromarray(img).resize(
        (W, H), Image.LANCZOS)  # [W,H 3] note!! resize的輸入是(W,H)
    img_dn = np.asarray(img_dn)  # [H,W,3]
    seg_score_dn = F.interpolate(torch.tensor(seg_score[np.newaxis, :, :, :]),
                                 size=(H, W),
                                 mode="bilinear",
                                 align_corners=False).squeeze().numpy()
    # [C,H,W]
    # seg_label_dn_ = F.interpolate(torch.tensor(seg_score[np.newaxis, :, :],
    #                                            dtype=torch.float64),
    #                               size=(H, W),
    #                               mode="bilinear",
    #                               align_corners=False).squeeze().numpy()
    # seg_label_dn = np.argmax(seg_label_dn_, axis=0)
    # include ignore label 255
    seg_label = seg_label.astype(np.uint8)
    # === using PIL to downsample
    seg_label_PIL_dn = Image.fromarray(seg_label).resize(
        (W, H), Image.NEAREST)  # [W,H 3] note!! resize的輸入是(W,H)
    seg_label_dn = np.asarray(seg_label_PIL_dn)  # [H,W]
    # === using torch.bilinear to downsample
    seg_label_torch_dn = seg_score_dn.argmax(axis=0)  # [H,W]
    # === vote
    seg_label_dn = np.where(seg_label_torch_dn == seg_label_PIL_dn,
                            seg_label_torch_dn, 255)
    if show_infromation:  # show infromation
        print("seg_label.shape ", seg_label.shape)
        print("np.unique(seg_label) ", np.unique(seg_label))
        print("np.unique(seg_label_dn) ", np.unique(seg_label_dn))
    # [H,W]

    # seg_score_dn = Image.fromarray(seg_score_chw).resize(
    #     (W, H), Image.LANCZOS)  # [H,W,3]
    # seg_label_dn = np.argmax(seg_score_dn, axis=2)
    # =======================================================================
    # FYI,By deepLabv2 , it say that
    # Image.resize can do better downsample than cv2 and torch interpolation
    # =======================================================================
    # ==== use torch to down sample
    # seg_label_dn = F.interpolate(torch.tensor(
    #     seg_label[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
    #                              size=(H, W),
    #                              mode="nearest").squeeze().numpy()
    # === just for visualization
    seg_label_dn_up = F.interpolate(torch.tensor(
        seg_label_dn[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
                                    size=(H_origin, W_origin),
                                    mode="nearest").squeeze().numpy()

    # seg_label_dn = np.argmax(seg_score_dn, axis=0)
    """ === save seg_label_dn_up in destination4visulization + '_DN_UP' === """
    scipy.misc.toimage(seg_label_dn_up,
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode='P').save(
                           os.path.join(destination4visulization + '_DN_UP',
                                        "{}.png".format(img_name)))

    def save_pseudo_label(seg_score,
                          seg_label,
                          destination,
                          img_name="2007_000032",
                          save_npy=True):
        """
        Save Label and Label Score to `.png` and dictionary
        ===
        - label would be upsample to save
        - `img_name`: str, only file name, not include extension or path
        - `seg_score`: numpy array, shape: [num_class,H,W] 
        - `seg_label`: numpy array, shape: [H,W] 
        """
        if not os.path.exists(destination):
            os.mkdir(destination)

        pseudo_label_dict = dict()
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        pseudo_label_dict[0] = seg_score[0]
        # key range from 0~20 if you use VOC dataset
        for key in img_label:  # img_label +1 = segmentation_label
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # pseudo_label_dict[int(key)] or pseudo_label_dict[int(key+1)] ???
            pseudo_label_dict[int(key + 1)] = seg_score[int(key + 1)]
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # save score
        if save_npy:
            destination_np = destination4logit
            if not os.path.exists(destination_np):
                os.mkdir(destination_np)
            np.save(os.path.join(destination_np, img_name), pseudo_label_dict)
        # Save label mask
        scipy.misc.toimage(seg_label,
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(destination,
                                            "{}.png".format(img_name)))

    """ === downsampling label for save in .png === """
    save_pseudo_label(seg_score=seg_score_dn,
                      seg_label=seg_label_dn,
                      img_name=img_name,
                      destination=destination4visulization + '_DN',
                      save_npy=False)
    """ === upsampling label for save in .png === """
    save_pseudo_label(seg_score=seg_score,
                      seg_label=seg_label,
                      img_name=img_name,
                      destination=destination4visulization + '_UP',
                      save_npy=False)


""" === 6.29 """


def refine_PPL_given_IRN_cam(img_name,
                             PPL_path=None,
                             irn_Bmap_path=None,
                             destination4visulization=None,
                             destination4logit=None,
                             show_infromation=False,
                             erase_radious=3,
                             BmapThred=.8):
    """
    Given the cam generate by IRNet, Generate partial pseudo label with ratio
    ---
    the data folder will be generated:
    - destination4visulization
    - destination4visulization + '_UP'
    """
    img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
    H_origin, W_origin, C = img.shape
    # print("img.shape", img.shape)
    H = int(np.ceil(H_origin / 8))
    W = int(np.ceil(W_origin / 8))
    """ ===  load upsample size Boundary map === """
    bMap = np.load(
        os.path.join(args.path4boundaryMap_logit + '_up',
                     img_name + '.npy'))  # [H,W]
    """=== generate pseudo label and idx for train,test ==="""
    PPL = Image.open(os.path.join(PPL_path, img_name + ".png"))
    PPL = np.array(PPL)  # [H_origin,W_origin]
    # === use ratio to be the threshold
    thred4Bun = sorted(bMap.flatten())
    thred4Bun = thred4Bun[int(len(thred4Bun) *
                              BmapThred)]  # 取低 --> 高的30%作為threshold
    # === use a fixed number to be the threshold
    # thred4Bun = .7
    mask = np.where(PPL == 255, 1, 0)
    new_PPL = np.copy(PPL)
    """=== erase class which cross boundary ==="""
    def boundary_clr(card_x, card_y, H, W, radius, PPL, new_PPL):
        if bMap[card_x, card_y] > thred4Bun:
            for idx_x in np.arange(card_x - radius, card_x + radius + 1):
                for idx_y in np.arange(card_y - radius, card_y + radius + 1):
                    if (0 < idx_x < H) and (0 < idx_y < W):
                        new_PPL[idx_x, idx_y] = 255

    def boundary_clr(card_x, card_y, H, W, radius, PPL, new_PPL):
        if bMap[card_x, card_y] > thred4Bun:
            for idx_x in np.arange(card_x - radius, card_x + radius + 1):
                for idx_y in np.arange(card_y - radius, card_y + radius + 1):
                    if (0 < idx_x < H) and (0 < idx_y < W):
                        new_PPL[idx_x, idx_y] = 255

    for i in range(H_origin):
        for j in range(W_origin):
            if PPL[i, j] == 255:
                boundary_clr(card_x=i,
                             card_y=j,
                             H=H_origin,
                             W=W_origin,
                             radius=erase_radious,
                             PPL=PPL,
                             new_PPL=new_PPL)

    # =================================
    # save CAM in dictionary
    # i think this part can be delete,
    # just set cams = np.load(.......)
    # =================================
    if not os.path.exists(os.path.join(destination4visulization + '_UP')):
        os.makedirs(os.path.join(destination4visulization + '_UP'))
    """ === save seg_label_dn_up in destination4visulization + '_UP' === """
    scipy.misc.toimage(new_PPL, cmin=0, cmax=255, pal=colors_map,
                       mode='P').save(
                           os.path.join(destination4visulization + '_UP',
                                        "{}.png".format(img_name)))

    def save_pseudo_label(seg_score,
                          seg_label,
                          destination,
                          img_name="2007_000032",
                          save_npy=True):
        """
        Save Label and Label Score to `.png` and dictionary
        ===
        - label would be upsample to save
        - `img_name`: str, only file name, not include extension or path
        - `seg_score`: numpy array, shape: [num_class,H,W] 
        - `seg_label`: numpy array, shape: [H,W] 
        """
        if not os.path.exists(destination):
            os.mkdir(destination)

        pseudo_label_dict = dict()
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        pseudo_label_dict[0] = seg_score[0]
        # key range from 0~20 if you use VOC dataset
        for key in img_label:  # img_label +1 = segmentation_label
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # pseudo_label_dict[int(key)] or pseudo_label_dict[int(key+1)] ???
            pseudo_label_dict[int(key + 1)] = seg_score[int(key + 1)]
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # save score
        if save_npy:
            destination_np = destination4logit
            if not os.path.exists(destination_np):
                os.mkdir(destination_np)
            np.save(os.path.join(destination_np, img_name), pseudo_label_dict)
        # Save label mask
        scipy.misc.toimage(seg_label,
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(destination,
                                            "{}.png".format(img_name)))

    """ === downsampling label for save in .png === """
    # === using PIL to downsample
    PPL_PIL_dn = Image.fromarray(new_PPL.astype(np.uint8)).resize(
        (W, H), Image.NEAREST)  # [W,H 3] note!! resize的輸入是(W,H)
    new_PPL_dn = np.asarray(PPL_PIL_dn)  # [H,W]
    if np.min(new_PPL_dn.flatten()) < 0:
        input("there are 0 in PPL...")
    if not os.path.exists(destination4visulization):
        os.makedirs(destination4visulization)
    scipy.misc.toimage(new_PPL_dn, cmin=0, cmax=255, pal=colors_map,
                       mode='P').save(
                           os.path.join(destination4visulization,
                                        "{}.png".format(img_name)))


if __name__ == "__main__":
    from os.path import join as opj
    """
    - data_RES(UP_CRF_DN): apply CRF in upsample(original size) CAM 
    then save label & score in down sample
    - data_RES(2020): apply CRF in downsample CAM
    then save label & score in upsample
    """
    # generate partial label >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    img_list = load_img_name_list(args.path4train_images)
    t_start = time.time()
    topredion_rate = .3

    cam_mode = "cam"
    if cam_mode == "irn":
        pred_folder = "{}@PIL_near@confident_ratio_{}_UP".format(
            opj(args.path4partial_label_label,
                "RES_CAM_TRAIN_AUG_PARTIAL_PSEUDO_LABEL"), topredion_rate)
        cam_folder = "PPL_mix_Bmap"
    elif cam_mode == "rw":
        pred_folder = "../psa/RES_RW_np"
        cam_folder = "RES_RW_np"
    else:
        pred_folder = "../psa/RES_CAM_TRAIN_AUG"
        cam_folder = "RES_CAM_TRAIN_AUG_PARTIAL_PSEUDO_LABEL"

    save_folder = "{}@PIL_near@confident_ratio_{}_{}".format(
        cam_folder, topredion_rate, cam_mode)

    for idx, img_name in enumerate(img_list, start=1):
        print("[{}/{}] {} time: {}m {}s".format(
            idx, np.size(img_list), img_name, (time.time() - t_start) // 60,
            int((time.time() - t_start) % 60)),
              end='\r')
        gen_partial_label_with_ratio(img_name,
                                     cam_mode=cam_mode,
                                     predict_root=pred_folder,
                                     destination4visulization=os.path.join(
                                         args.path4partial_label_label,
                                         save_folder),
                                     destination4logit=os.path.join(
                                         args.path4partial_label_logit,
                                         save_folder),
                                     confident_region=topredion_rate)
        # refine_PPL_given_IRN_cam(img_name,
        #                          irn_Bmap_path=args.path4boundaryMap + "_up",
        #                          PPL_path=pred_folder,
        #                          destination4visulization=os.path.join(
        #                              args.path4partial_label_label,
        #                              save_folder),
        #                          destination4logit=os.path.join(
        #                              args.path4partial_label_logit,
        #                              save_folder),
        #                          erase_radious=3,
        #                          BmapThred=.7)
    print("")
    from utils import evaluate_dataset_IoU
    evaluate_dataset_IoU(
        predicted_folder=os.path.join(args.path4partial_label_label,
                                      save_folder + "_UP"),
        descript="exp: use Bmap erase PPL boundary@bunThred=top80 radious=3")
