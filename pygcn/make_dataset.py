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


if __name__ == "__main__":
    """
    - data_RES(UP_CRF_DN): apply CRF in upsample(original size) CAM 
    then save label & score in down sample
    - data_RES(2020): apply CRF in downsample CAM
    then save label & score in upsample
    """
    # gen_dataset(predict_root=os.path.join("..", "psa", "RES_CAM__"),
    #             destination="data_RES(UP_CRF_DN)",
    #             destination4visulization="RES_PSEUDO_LABEL")
    # args.path4train_images = os.path.join("..", "psa", "voc12", "val.txt")
    # args.path4PsaFeature = os.path.join(
    #     "..", "psa", "AFF_FEATURE_VGG")  # "AFF_FEATURE_VGG" | "AFF_FEATURE_2020" | "aff_map"
    # gen_dataset(predict_root=os.path.join("..", "psa", "RES_CAM_TRAIN_AUG"),
    #             destination="data_v8",
    #             destination4visulization="RES_CAM_TRAIN_AUG_v8",
    #             img_list_path=args.path4train_aug_images)

    # generate partial label >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    img_list = load_img_name_list(args.path4train_images)
    t_start = time.time()
    for idx, img_name in enumerate(img_list, start=1):
        print("[{}/{}] {} time: {}m {}s".format(
            idx, np.size(img_list), img_name, (time.time() - t_start) // 60,
            int((time.time() - t_start) % 60)))
        gen_partial_label(
            img_name,
            predict_root=
            "../psa/RES_CAM_TRAIN_AUG",
            destination=
            "../psa/RES_CAM_TRAIN_AUG_partial_pseudo_label",
            destination4visulization=
            "../psa/RES_CAM_TRAIN_AUG_PARTIAL_PSEUDO_LABEL"
        )
