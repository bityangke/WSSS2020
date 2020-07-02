import numpy as np
import scipy.sparse as sp
import torch

import pickle as pkl
import networkx as nx
import torch.nn.functional as F
from torch import nn
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy.misc
import sys
import os
from config import opt as args
from PIL import Image
import datetime
import fire
import time
import getpass
from cv2 import imread, imwrite
import pydensecrf.densecrf as dcrf

colors_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [0, 0, 255]]
IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

SEG_LIST = [
    'BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# transfer segmentation_class_name <-> segmentation_class_ID
SEG_NAME_TO_ID = dict(zip(SEG_LIST, range(len(SEG_LIST))))
SEG_ID_TO_NAME = dict(zip(np.arange(len(SEG_LIST)), SEG_LIST))

# transfer image_class_name <-> image__class_ID
CLS_NAME_TO_ID = dict(zip(SEG_LIST[1:], range(len(SEG_LIST[1:]))))
CLS_ID_TO_NAME = dict(zip(np.arange(len(SEG_LIST[1:])), SEG_LIST[1:]))


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {
        c: np.identity(len(classes))[i, :]
        for i, c in enumerate(classes)
    }
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def Laplacian(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


"""2020.4"""


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    IoU = IOUMetric(num_class)
    IoU.addbatch(predictions,groundtruths)
    acc, acc_cls, iu, mean_iu, fwavacc = IoU.evaluate()
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.num_pixel4dataset = 0
        self.num_train_pixel4dataset = 0

    def _fast_hist(self, label_pred, label_true):
        """
        - 回傳confussion matrix
        - np.bincount(X,minlength) 回傳 cufussion matrix (kxk)
        - X = [k,k] integer matrix
        - minlength用來指定bin的數目
        - 如果不指定minlength 此函數會選X中的最大值+1作為bin的數目
        - e.g. X=[1,1,1,2,1,3,3,3] => np.bincount(X) return [0,4,1,3]
        - e.g. X=[1,1,1,1,1,3,3,3] => np.bincount(X) return [0,5,0,3]
        """
        # 忽略255 and 負值
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # 不只GT 連prediction的255也要ignore
        mask = mask & (label_pred >= 0) & (label_pred < self.num_classes)
        # if you use sum(mask) and len(size), it take 30x time than numpy
        self.num_train_pixel4dataset += np.sum(mask)
        self.num_pixel4dataset += mask.size
        """
            以下是把true label放到num_class進位,predict放在個位數
            因此ij表示true=i,predict=j
            接著bincount會計算每個ij的數量,由於是以index的方式存放,所以是一維tensor.要拉回2維表示
            """
        # print("len(np.unique) ", np.unique(label_true[mask].astype(int)))
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes**2).reshape(self.num_classes,
                                                   self.num_classes)

        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        """
        iu: [num_classes,] is a numpy array. each item is IoU for class_i
        mean_iu_tensor: a tensor, take the average of iu
        acc_cls: accuracy for each class
        """
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) +
                                   self.hist.sum(axis=0) - np.diag(self.hist))
        # nanmean just ignore nan in the item
        mean_iu_tensor = torch.from_numpy(np.asarray(np.nanmean(iu)))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu_tensor, fwavacc


def load_img_name_list(dataset_path):
    """
    return imgs_list e.g.  imgs_list[0] = 2007_000121
    """
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [
        img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list
    ]
    """ /JPEGImages/2007_000121.jpg -> [-15:-4] = 2007_000121 """
    return img_name_list


def evaluate_dataset_IoU(predicted_folder=args.path4save_img,
                         path4GT=args.path4Class,
                         file_list=args.path4train_images,
                         ignore_img_list=[],
                         save_info=True,
                         descript=None):
    """
    Evaluate the Prediction in predicted_folder
    ===
    - only file appear in the file_list would be calculate
    - assum the file is `.png`
    - descript: a message to record in `information_for_IoU.md`
    ---
    Example of Use
    ---
    - t_start = time.time()
    - predict_folder = `"..\RES_CAM_label(train)"`
    - evaluate_dataset_IoU(predicted_folder=predict_folder)
    - show_timing(time_start=t_start, time_end=time.time())
    """

    img_list = load_img_name_list(file_list)  # e.g. 2007_000032
    IoU = IOUMetric(args.num_class)
    num_imgs = len(img_list)
    i = 0
    mask_predit_batch = []
    mask_GT_batch = []
    for img_name in img_list:
        if img_name in ignore_img_list:
            print("ignore: {}".format(img_name))
            continue
        i = i + 1
        print("[{}/{}]evaluate: ".format(i, num_imgs), img_name, end='\r')
        mask_gt = Image.open(os.path.join(path4GT, img_name + '.png'))
        mask_gt = np.asarray(mask_gt)
        mask_predit = Image.open(
            os.path.join(predicted_folder, img_name + '.png'))
        mask_predit = np.asarray(mask_predit)
        # upsampling
        if mask_predit.shape[0] < mask_gt.shape[0] or mask_predit.shape[
                1] < mask_gt.shape[1]:
            mask_predit_up = Image.fromarray(mask_predit).resize(
                (mask_gt.shape[1], mask_gt.shape[0]), Image.NEAREST)
            mask_predit = np.asarray(mask_predit_up)
        # mask_predit = np.where(mask_gt == 255, mask_gt, mask_predit)
        mask_predit_batch.append(mask_predit)
        mask_GT_batch.append(mask_gt)
    IoU.add_batch(mask_predit_batch, mask_GT_batch)
    acc, acc_cls, iu, mean_iu_tensor, fwavacc = IoU.evaluate()

    # show information
    print("pseudo pixel label ratio: {:>5.2f} %".format(
        IoU.num_train_pixel4dataset / IoU.num_pixel4dataset * 100))
    # show IoU of each class
    print("=" * 34)
    for idx, iu_class in enumerate(iu):
        print("{:12}: {:>17.2f} %".format(SEG_ID_TO_NAME[idx], iu_class * 100))
    print("=" * 34)
    print("IoU:{:>27.2f} %  Acc:{:>13.2f} %".format(mean_iu_tensor * 100,
                                                    acc * 100))
    print("=" * 34)
    # === save information in `information_for_IoU.md`
    if descript is not None:
        time_now = datetime.datetime.today()
        time_now = "{}-{}-{}  {}:{}".format(time_now.year, time_now.month,
                                            time_now.day, time_now.hour,
                                            time_now.minute)

        if not os.path.isfile("meanIoU@Jun.md"):
            f = open("meanIoU@Jun.md", "w")
            f.close()

        with open("meanIoU@Jun.md", "r") as f:
            old_context = f.read()
        with open("meanIoU@Jun.md", "r+") as f:
            f.write("{}  \n".format(time_now))
            f.write("---\n")
            f.write("\n|Setting|Value|\n")
            f.write("|-|-|\n")
            f.write("**predicted folder**|{}  \n".format(predicted_folder))
            f.write("**Dataset files** |{}  \n".format(
                os.path.basename(args.path4data)))
            f.write("**AFF_path**| {}  \n".format(
                os.path.basename(args.path4AffGraph)))
            f.write("**apply CRF**| {}  \n".format(args.use_crf))
            f.write("**epoch**| {}  \n".format(args.max_epoch))
            f.write("**hid unit**| {}  \n".format(args.num_hid_unit))
            f.write("**drop out**| {}  \n".format(args.drop_rate))
            f.write("**stript**| {}  \n".format(descript))
            f.write("-" * 3)
            f.write("\n|Class|IoU|\n")
            f.write("|-|-|\n")
            for idx, iu_class in enumerate(iu):
                f.write("{:12}| {:>17.2f} %  \n".format(
                    SEG_ID_TO_NAME[idx], iu_class * 100))
            f.write("-" * 3)
            f.write("\n|pseudo pixel label ratio|Acc|meanIoU|\n")
            f.write("|-|-|-|\n")
            f.write("|{:<5.2f} %|{:>5.2f} % | {:>27.2f} % \n".format(
                IoU.num_train_pixel4dataset / IoU.num_pixel4dataset * 100,
                acc * 100,
                mean_iu_tensor.item() * 100))
            f.write("-" * 3 + "\n")
            f.write(old_context)

    return mean_iu_tensor.item(), acc


"""2020.5.17"""


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    """
    this function haven't use
    ---
    - img: np_array [h,w,c]
    - probs: prediction_score [c,h,w]
    - t: number of iteration for inference

    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80 / scale_factor,
                           srgb=13,
                           rgbim=np.copy(img_c),
                           compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def compute_joint_loss(ori_img, seg, seg_label, croppings, critersion,
                       DenseEnergyLosslayer):
    """
    1. seg_label: pseudo label for segmentation\n
    2. seg: ouput of seg_model\n (b,c,w,h)
    ---\n
    seperate bg_loss and fg_loss
    return cross-entropy loss, dense loss
    """
    seg_label = np.expand_dims(
        seg_label, axis=1
    )  # expand channel dimension, because label only have one channel
    seg_label = torch.from_numpy(seg_label)  # to tensor

    w = seg_label.shape[2]
    h = seg_label.shape[3]
    # upsampling or down sampling
    pred = F.interpolate(seg, (w, h), mode="bilinear", align_corners=False)

    # apply softmax to model prediction
    pred_softmax = torch.nn.Softmax(dim=1)
    pred_probs = pred_softmax(pred)

    # numpy array to tensor
    ori_img = torch.from_numpy(ori_img.astype(np.float32))
    croppings = torch.from_numpy(
        croppings.astype(np.float32).transpose(2, 0, 1))

    # put pseudo label in GPU
    seg_label_tensor = seg_label.long().cuda()

    seg_label_copy = torch.squeeze(seg_label_tensor.clone())
    bg_label = seg_label_copy.clone()
    fg_label = seg_label_copy.clone()

    # put foreground ignore(255 mean ignore)
    bg_label[seg_label_copy != 0] = 255
    # put background ignore(255 mean ignore)
    fg_label[seg_label_copy == 0] = 255

    bg_celoss = critersion(pred, bg_label.long().cuda())
    fg_celoss = critersion(pred, fg_label.long().cuda())

    # dense loss
    dloss = DenseEnergyLosslayer(ori_img, pred_probs, croppings, seg_label)
    dloss = dloss.cuda()

    celoss = bg_celoss + fg_celoss

    return celoss, dloss


class Normalize():
    """    
    Normalize使用範例

    train_dataset = voc12.data.VOC12ClsDataset(
    args.train_list,
    voc12_root=args.voc12_root,
    transform=transforms.Compose([
        imutils.RandomResizeLong(256, 512),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3,
                                contrast=0.3,
                                saturation=0.3,
                                hue=0.1), np.asarray
    ]),
    transform2=imutils.Compose([
        imutils.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
        imutils.RandomCrop(args.crop_size), imutils.HWC_to_CHW
    ]))

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True)
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img, ori_img, croppings):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]
        croppings = np.ones_like(imgarr)
        return proc_img, imgarr, croppings


def _crf_with_alpha(ori_img, cam_dict, alpha=32, use_crf=True):
    """
    - return CRF( CAM_plus_bg )
    ===
    - ori_img: numpy array | shape=[H,W,C] | img = imread(img_name)
    - alpha: integer (usually 4 or 32)
    - bg_score is compute by
        - v = np.array(list(cam_dict.values()))
        - bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    """
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    if use_crf:
        crf_score = crf_inference(ori_img,
                                  bgcam_score,
                                  labels=bgcam_score.shape[0])
        bgcam_score = crf_score

    n_crf_al = np.zeros([args.num_class, bg_score.shape[1], bg_score.shape[2]])

    n_crf_al[0, :, :] = bgcam_score[0, :, :]

    # transfer array idx to 21 class instead of e.g. 3, because crf_score do not keep the corrected idx except idx "0"
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = bgcam_score[i + 1]

    return n_crf_al


def load_image_label_from_xml(img_name, voc12_root):
    """
    No background index be consider
    ===
    - img_name: e.g. 2007_000032
    - return np array lenth=20
    """
    from xml.dom import minidom

    el_list = minidom.parse(
        os.path.join(voc12_root, ANNOT_FOLDER_NAME,
                     img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in SEG_LIST:
            cat_num = CLS_NAME_TO_ID[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def compute_seg_label(ori_img,
                      cam_label,
                      norm_cam,
                      threshold4conf=0.8,
                      use_crf=True,
                      confident_region=1.):
    """
    :norm_cam: value between 0,1]
    ---
    :let seg_score_np = CRF(ha_CAM+la_CAM)
    :return 
        - mask of seg_score with ignore region (i.e. ignore label=255)
        - seg_score_np
    """
    cam_label = cam_label.astype(np.uint8)
    cam_dict = {}
    cam_np = np.zeros_like(norm_cam)
    for i in range(20):
        if cam_label[i] > 1e-5:
            cam_dict[i] = norm_cam[i]
            cam_np[i] = norm_cam[i]

    bg_score = np.power(1 - np.max(cam_np, 0), 32)  # confident BG
    bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, cam_np))
    _, bg_w, bg_h = bg_score.shape

    cam_img = np.argmax(cam_all, 0)

    # === 計算 CRF(ha_CAM+la_CAM) and 引入ignore value 255
    # condident FG
    crf_la = _crf_with_alpha(ori_img, cam_dict, 4, use_crf=use_crf)
    # condident BG
    crf_ha = _crf_with_alpha(ori_img, cam_dict, 32, use_crf=use_crf)
    crf_la_label = np.argmax(crf_la, 0)
    crf_ha_label = np.argmax(crf_ha, 0)
    crf_label = crf_la_label.copy()
    crf_label[crf_la_label == 0] = 255  # ignore BG, 因為BG不夠準要拿ha的
    crf_label[crf_ha_label == 0] = 0  # 因為BG不夠準要拿ha的BG

    # === 計算 cam_not_sure_region, for each class CAM, find the top 60% score region as sure region and union all class sure region
    single_img_classes = np.unique(crf_la_label)
    cam_sure_region = np.zeros([bg_w, bg_h], dtype=bool)
    for class_i in single_img_classes:  # for each class CAM do...
        if class_i != 0:  # class is foreground
            class_not_region = (cam_img != class_i)  # [H,W]
            cam_class = cam_all[class_i, :, :]  # CAM for class_i [H,W]
            cam_class[class_not_region] = 0  # not sure region value = 0
            cam_class_order = cam_class[
                cam_class >
                0.1]  # only keep the CAM value that the prediction score > 0.1

            # === sort the prediction score [H_thre x W_thre]
            # === W_thre means that the CAM score in this idx is higher than a threshold (in this case = 0.1)
            cam_class_order = np.sort(cam_class_order)
            # set the 60% idx
            confidence_pos = int(cam_class_order.shape[0] *
                                 (1. - confident_region))
            # take the top 40% value as the threshold
            confidence_value = cam_class_order[confidence_pos]

            # keep the value which higher than the threshold
            class_sure_region = (cam_class > confidence_value)
            # expand the sure region
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
        else:  # class is background
            # find FG region
            class_not_region = (cam_img != class_i)
            # take background map. (cam_all means CAM+BG_map)
            cam_class = cam_all[class_i, :, :]
            # make FG region score = 0
            cam_class[class_not_region] = 0
            # only confident background region would be take
            class_sure_region = (cam_class > threshold4conf)
            # expand the sure region
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)

    cam_not_sure_region = ~cam_sure_region

    #  拿la的前景score and 拿ha的背景score 做結合
    crf_label_np = np.concatenate(
        [np.expand_dims(crf_ha[0, :, :], axis=0), crf_la[1:, :, :]])

    crf_not_sure_region = np.max(crf_label_np, 0) < threshold4conf
    not_sure_region = np.logical_or(crf_not_sure_region, cam_not_sure_region)

    crf_label[not_sure_region] = 255

    return crf_label, crf_label_np


def gen_label(num_class=20,
              save_img=True,
              voc_imgs_root=args.path4Image,
              predict_root=args.path4CAM,
              destination=args.path4Pseudo_label,
              use_crf=False,
              save_npy=False):
    """
        - depend function
            - compute_seg_label()
            - _crf_with_alpha()
        ---
        - 對predict_root裡的.npy計算CAM_ha, CAM_la得confident FG and BG score, 再concatenate得 all_CAM. 
        - predict_root 裡包含.npy的CAM 只有20類別(不含BG),且是以dictionary存放
        - return CRF(ha_CAM+la_CAM) with ignore region (i.e. ignore label=255) and 儲存CRF score & msk
        ---
        Example of Use
        ---
        - alpha = `4
        - dataset = `"train"`
        - predict_folder = os.path.join(`".."`, `"RES38_HALA_PSEUDO_LABEL(2020)"`)
        - gen_label(predict_root=`r"..\psa\RES_CAM__"`,
        -      destination=predict_folder,
        -      use_crf=`False`)
        """
    # load files' name as list (not include path)
    destination_np = destination + "_SCORE"
    imgs = [
        os.path.splitext(f)[0] for f in os.listdir(predict_root)
        if os.path.splitext(f)[-1] == '.npy'
    ]
    # load files' name as list (absolute path)
    voc_imgs = [
        os.path.join(voc_imgs_root, f) for f in os.listdir(voc_imgs_root)
        if os.path.splitext(f)[0] in imgs
    ]

    if not os.path.exists(destination):
        os.mkdir(destination)
    if not os.path.exists(destination_np):
        os.mkdir(destination_np)
    lenth = len(imgs)
    # 已經儲存過的就別再算了
    cam_exist_list = [os.path.splitext(f)[0] for f in os.listdir(destination)]
    for idx, (cam_file, img_name) in enumerate(zip(imgs, voc_imgs)):
        if cam_file in cam_exist_list:
            print("[{}/{}]{} already exist!!".format(idx + 1, lenth, cam_file))
            continue
        img = imread(img_name)
        (H, W) = img.shape[:2]
        cams = np.zeros((num_class, H, W))
        dict_np = np.load(os.path.join(predict_root, cam_file + '.npy')).item()
        print('[{idx}/{lenth}] cam_file:{cam_file}  cams.shape: {cams_shape}'.
              format(idx=idx + 1,
                     lenth=lenth,
                     cam_file=cam_file,
                     cams_shape=cams.shape))
        cam_label = np.zeros(20)
        cam_dict = {}
        cam_temp = None
        for key, cam in dict_np.items():
            cam = F.interpolate(torch.tensor(cam[np.newaxis,
                                                 np.newaxis, :, :]), (H, W),
                                mode="bilinear",
                                align_corners=False).numpy()
            cams[key] = cam  # note! cam label from 0-29
            cam_temp = cam.copy()
            cam_label[key] = 1
            cam_dict[key] = cams[key]

        print(
            '[{idx}/{lenth}] cam_file:{cam_file}  cams.shape: {cams_shape}  cam.shape: {cam_shape}'
            .format(idx=idx + 1,
                    lenth=lenth,
                    cam_file=cam_file,
                    cams_shape=cams.shape,
                    cam_shape=cam_temp.shape))

        cam_label = cam_label.astype(np.uint8)
        seg_label_crf_conf, seg_score_crf_conf = compute_seg_label(
            ori_img=img, cam_label=cam_label, norm_cam=cams, use_crf=use_crf)

        # === Save label score by dictionary
        pseudo_label_dict = dict()
        img_label = load_image_label_from_xml(cam_file)
        pseudo_label_dict[0] = seg_score_crf_conf[0]
        # key range from 0~20 if you use VOC dataset
        for key in img_label:
            pseudo_label_dict[int(key)] = seg_score_crf_conf[int(key + 1)]

        if save_npy:
            np.save(os.path.join(destination_np, "{}".format(cam_file)),
                    pseudo_label_dict)
        # Save label mask
        scipy.misc.toimage(seg_label_crf_conf,
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(destination, cam_file + '.png'))


def show_timing(time_start, time_end, show=False):
    """
    Show timimg in the format: h m s
    ===
    Example of Use
    ---
    - t_start = time.time()  
    - you code/function you want to timing  
    - show_timing(t_start,time.time())
    """
    time_hms = "Total time elapsed: {:.0f} h {:.0f} m {:.0f} s".format(
        (time_end - time_start) // 3600, (time_end - time_start) / 60 % 60,
        (time_end - time_start) % 60)
    if show:
        print(time_hms)
    return time_hms


def visulize_cam(root, destination="VGG_CAM_CRF", alpha=16, use_crf=False):
    """
    - Transfer `.npy` CAM into pseudo label
    ===

    Example of Use
    ---
    - t_start = time.time()
    - cam_folder = "RES_CAM__"
    - alpha = 16
    - visulize_cam(root=os.path.join("..", "psa", cam_folder),
                  destination="{_VIS}".format(cam_folder),
                  alpha=alpha)
    - show_timing(time_start=t_start, time_end=time.time())
    """
    destination = "{}_HA{}".format(destination, alpha)
    # predict_folder = os.path.join(
    #     "..", "VGG_CAM_LA_VIS"
    # )  # "VGG_CAM_PSEUDO_LABEL(54p63)HW_UP") # "RES_CRF_CAM_HA32(train)") #"..\\train_label"  # "RES38_PSEUDO_LABEL"
    # evaluate_dataset_IoU(predicted_folder=predict_folder)
    if not os.path.exists(destination):
        os.mkdir(destination)
    cam_path = root
    train_list = np.array(load_img_name_list(args.path4train_images))
    len_list = train_list.size
    for idx, name in enumerate(train_list):
        print("[{}/{}] {}...".format(idx + 1, len_list, name), end='\r')
        cam_dict = np.load(os.path.join(cam_path, name + '.npy'),
                           allow_pickle=True).item()
        img = imread(os.path.join(args.path4Image, name + '.jpg'))
        pseudo_label = _crf_with_alpha(ori_img=img,
                                       cam_dict=cam_dict,
                                       alpha=alpha,
                                       use_crf=use_crf)
        scipy.misc.toimage(np.argmax(pseudo_label, axis=0),
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(destination, name + '.png'))

    print("spend time:{:.1f}s".format(time.time() - t_start))


def crf_with_alpha_4prediction(ori_img, cam_dict, use_crf=True):
    """
    - return CRF( CAM_plus_bg ) [21,H,W]
    ===
    - ori_img: np array, shape: [H,W,3]
    - cam_dict: CAM, dictionary, each value is CAM map corresponding to the class 
    - bg_score:
        - v = np.array(list(cam_dict.values()))
        - bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    """
    v = np.array(list(cam_dict.values()))
    if use_crf:
        crf_score = crf_inference(ori_img, v, labels=v.shape[0])

    n_crf_al = np.zeros(
        [args.num_class, crf_score.shape[1], crf_score.shape[2]])
    # transfer array idx to 21 class instead of e.g. 3, because crf_score do not keep the corrected idx except idx "0"
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key] = crf_score[i]

    return n_crf_al


def crf_inference_inf(img_name,
                      img=None,
                      path4CRFLabel=None,
                      probs=None,
                      prediction_root=None,
                      t=10,
                      scale_factor=1,
                      labels=21):
    """
    apply CRF to model prediction and save th result in `args.path4prediction_np`
    ===
    - img:  ,shape: [H,W,3]
    - probs: ,shape: [C,H,W], it's value represent log(p)
    - default save path:args.path4saveCRF
    - path4CRFLabel: path to save CRF prediction as label
    """
    from pydensecrf.utils import unary_from_softmax
    print("apply CRF...")
    path4CRFLabel = args.path4saveCRF
    # === read_image ===
    # img_label = load_image_label_from_xml(img_name=img_name,
    #                                       voc12_root=args.path4VOC_root)
    if img is None:
        img = imread(os.path.join(args.path4Image,
                                  img_name + '.jpg'))  # [H,W,3]
    H, W = img.shape[:2]
    # === after softmax, do CRF and return prediction distribution
    pred_softmax = torch.nn.Softmax(dim=0)
    # === generate probs by load model prediction in dict to numpy and assign it ===
    if probs is None:
        # === load predict_dict ===
        if prediction_root is None:
            prediction_root = os.path.join("predict_result_matrix_visual_new",
                                           "250")
        predict_np = np.zeros(shape=(args.num_class, H, W))
        prect_dict = np.load(os.path.join(prediction_root, img_name + '.npy'),
                             allow_pickle=True).item()
        # === transfer to array [21,H,W] ===
        for key in prect_dict.keys():
            print("key: ID: {}, name: {} ".format(key, SEG_ID_TO_NAME[key]))
            predict_np[key] = prect_dict[key]
    probs = pred_softmax(torch.Tensor(predict_np)).numpy()
    # === apply CRF
    n_labels = labels
    print("H,W:", img.shape[:2])
    print("predict_np.shape ", predict_np.shape)
    d = dcrf.DenseCRF2D(W, H, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=4 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83 / scale_factor,
                           srgb=5,
                           rgbim=np.copy(img_c),
                           compat=3)
    #=== setting from psa
    # d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=80 / scale_factor,
    #                        srgb=13,
    #                        rgbim=np.copy(img_c),
    #                        compat=10)
    Q = d.inference(t)
    crf_predict = np.array(Q).reshape((n_labels, H, W))
    if not os.path.exists(args.path4saveCRF):
        os.mkdir(path4CRFLabel)
    # === save img in s.path.join(args.path4saveCRF, img_name + '.png')
    scipy.misc.toimage(np.argmax(crf_predict, 0),
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode='P').save(
                           os.path.join(path4CRFLabel, img_name + '.png'))

    print("{} is save. for apply CRF".format(
        os.path.join(path4CRFLabel, img_name + '.png')))


def crf_inference_psa(img, probs, CRF_parameter, scale_factor=1, labels=21):
    """
    this setting is different from PSA

    IoU=62.44
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)
    pred_softmax = torch.nn.Softmax(dim=0)
    probs = pred_softmax(torch.tensor(probs)).numpy()
    # probs = pred_softmax(torch.from_numpy(probs).float()).numpy()
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    # =======================================================
    # === setting in PSA for la_CRF,ha_CRF
    # d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=80 / scale_factor,
    #                        srgb=13,
    #                        rgbim=np.copy(img),
    #                        compat=10)
    # =======================================================
    # setting in deeplab
    # CRF_parameter = args.CRF
    # CRF_parameter = args.CRF_psa
    d.addPairwiseGaussian(sxy=CRF_parameter["pos_xy_std"] / scale_factor,
                          compat=CRF_parameter["pos_w"])
    d.addPairwiseBilateral(sxy=CRF_parameter["bi_xy_std"] / scale_factor,
                           srgb=CRF_parameter["bi_rgb_std"],
                           rgbim=np.copy(img),
                           compat=CRF_parameter["bi_w"])
    t = CRF_parameter["iter_max"]
    # =======================================================
    # d.addPairwiseGaussian(sxy=4 / scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=83 / scale_factor,
    #                        srgb=5,
    #                        rgbim=np.copy(img),
    #                        compat=3)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def crf_psa(img_name,
            img=None,
            path4CRFLabel=None,
            probs=None,
            prediction_root=None,
            t=10,
            scale_factor=1,
            labels=21,
            save_path="PREDIC_CRF_PSA"):
    """
    A new setting from PSA
    ---
    still testing
    """
    if img is None:
        img = imread(os.path.join(args.path4Image,
                                  img_name + '.jpg'))  # [H,W,3]
    H, W = img.shape[:2]
    # === load predict_dict ===
    if prediction_root is None:
        prediction_root = os.path.join("predict_result_matrix_visual_new",
                                       "250")
    # predict_np = np.zeros(shape=(args.num_class, H, W))
    prect_dict = np.load(os.path.join(prediction_root, img_name + '.npy'),
                         allow_pickle=True).item()
    """ ===3. save random walk prediction as np array in RW_prediction  === """

    # >>>> save in dictionary() >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def rw_crf(predicted_dict, name=None):
        """
        - orig_img: [H,W,3], np array
        - predicted_dict: dictionary, each item is a [H,W] predicted score for coresponding class
        """
        v = np.array(list(predicted_dict.values()))
        # orig_img = orig_img.data.cpu().numpy().transpose((1, 2, 0))  # [H,W,3]
        img_path = os.path.join(args.path4Image, name + '.jpg')
        orig_img = np.asarray(Image.open(img_path))
        # === note that orig_img have the shape [H,W,3]
        crf_score = crf_inference_psa(orig_img, v, labels=v.shape[0])
        h, w = orig_img.shape[:2]
        crf_dict = dict()
        crf_score_np = np.zeros(shape=(args.num_class, h, w))
        for i, key in enumerate(predicted_dict.keys()):
            crf_score_np[key] = crf_score[i]
            crf_dict[key] = crf_score[i]
        return crf_score_np, crf_dict

    # === note that orig_img must be in shape [H,W,3]
    rw_crf_resut, crf_dict = rw_crf(predicted_dict=prect_dict, name=img_name)
    # === save as dictionary
    if not os.path.exists(save_path + "_np"):
        os.mkdir(save_path + "_np")
    np.save(os.path.join(save_path + "_np", img_name + '.npy'), crf_dict)
    """ ===4. save the random walk prediction as label in args.out_rw as .png === """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    scipy.misc.toimage(rw_crf_resut.argmax(axis=0),
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode="P").save(
                           os.path.join(save_path, img_name + '.png'))


def crf_deepLab(img_name,
                CRF_parameter,
                img=None,
                path4CRFLabel=None,
                probs=None,
                prediction_root=None,
                t=10,
                scale_factor=1,
                labels=21,
                save_path="PREDIC_CRF_PSA"):
    """
    A new setting from PSA
    ---
    still testing
    """
    if img is None:
        img = imread(os.path.join(args.path4Image,
                                  img_name + '.jpg'))  # [H,W,3]
    H, W = img.shape[:2]
    # === load predict_dict ===
    if prediction_root is None:
        prediction_root = os.path.join("predict_result_matrix_visual_new",
                                       "250")
    output = np.load(os.path.join(prediction_root, img_name + '.npy'),
                     allow_pickle=True)
    size = img.shape[:2]
    interp = nn.Upsample(size=(505, 505), mode='bilinear',
                         align_corners=True)  # [1,21,H_dn,W_dn]
    logits_t = interp(torch.tensor(output)).squeeze(dim=0)  # [21,505,505]
    logits = logits_t[:, :size[0], :size[1]]  # === [21,H,W]
    pred_seg_class = logits.argmax(dim=0).view(-1).long()  # === [H,W]
    appear_class = torch.unique(pred_seg_class)
    prect_dict = dict()
    # key range from 0~20 if you use VOC dataset
    for key in appear_class:  # img_label +1 = segmentation_label
        prect_dict[key] = logits[key].numpy()
    """ ===3. save random walk prediction as np array in RW_prediction  === """

    # >>>> save in dictionary() >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def rw_crf(predicted_dict, name=None):
        """
        - orig_img: [H,W,3], np array
        - predicted_dict: dictionary, each item is a [H,W] predicted score for coresponding class
        """
        v = np.array(list(predicted_dict.values()))
        # orig_img = orig_img.data.cpu().numpy().transpose((1, 2, 0))  # [H,W,3]
        img_path = os.path.join(args.path4Image, name + '.jpg')
        orig_img = np.asarray(Image.open(img_path))
        # === note that orig_img have the shape [H,W,3]
        crf_score = crf_inference_psa(orig_img,
                                      v,
                                      labels=v.shape[0],
                                      CRF_parameter=CRF_parameter)
        h, w = orig_img.shape[:2]
        crf_dict = dict()
        crf_score_np = np.zeros(shape=(args.num_class, h, w))
        for i, key in enumerate(predicted_dict.keys()):
            crf_score_np[key] = crf_score[i]
            crf_dict[key] = crf_score[i]
        return crf_score_np, crf_dict

    # >>>>>>>>>>>>>>>>>>>>>>>>>
    # === note that orig_img must be in shape [H,W,3]
    rw_crf_result, crf_dict = rw_crf(predicted_dict=prect_dict, name=img_name)
    # === save as dictionary
    if not os.path.exists(save_path + "_np"):
        os.mkdir(save_path + "_np")
    np.save(os.path.join(save_path + "_np", img_name + '.npy'), crf_dict)
    """ ===4. save the random walk prediction as label in args.out_rw as .png === """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    scipy.misc.toimage(rw_crf_result.argmax(axis=0),
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode="P").save(
                           os.path.join(save_path, img_name + '.png'))
    # >>>>>>>>>>>>>>>>>>>>>>>>>

    path_pred = os.path.join("pred_semi_VCIP")
    if not os.path.exists(path_pred):
        os.makedirs(path_pred)
    scipy.misc.toimage(logits.argmax(dim=0).numpy(),
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode="P").save(
                           os.path.join(path_pred, img_name + '.png'))


class HLoss(nn.Module):
    def __init__(self):
        """
        Example of Use:
        ---
        - criterion = HLoss()  
        - x = Variable(torch.randn(10, 10))
        - w = Variable(torch.randn(10, 3), requires_grad=True)
        - output = torch.matmul(x, w)
        - loss = criterion(output,gt_label,ignore_index=255)
        - loss.backward()
        - print(w.grad)
        """
        super(HLoss, self).__init__()

    def forward(self, x, gt, ignore_index=255):
        # b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -torch.exp(x) * x
        b = b[gt == 255].mean()  # sum all elements
        return b


class symmetricLoss(nn.Module):
    def __init__(self):
        """
        Example of Use:
        ---
        - criterion = HLoss()  
        - x = Variable(torch.randn(10, 10))
        - w = Variable(torch.randn(10, 3), requires_grad=True)
        - output = torch.matmul(x, w)
        - loss = criterion(output,gt_label,ignore_index=255)
        - loss.backward()
        - print(w.grad)
        """
        super(symmetricLoss, self).__init__()

    def forward(self, x, gt, ignore_index=255):
        one_hot_gt = torch.zeros_like(x)
        print("one_hot_gt.shape ", one_hot_gt.shape)
        print("gt.shape", gt.shape)
        gt_ = torch.where(gt != 255, gt, torch.tensor(0).cuda())
        input("[906]------>")
        one_hot_gt[np.arange(gt.size()[0]), gt_] = 1
        plogq = F.softmax(one_hot_gt + 1e-5, dim=1) * F.log_softmax(x, dim=1)
        qlogp = F.softmax(x, dim=1) * F.log_softmax(one_hot_gt + 1e-5, dim=1)
        b = -1.0 * (qlogp[gt != 255] +
                    plogq[gt != 255]).mean()  # sum all elements
        return b


def PPL_refine(img, Pselabel):
    """
    Refine Partial Pseudo Label with displacement
    ---
    - img,numpy array [H,W,3]
    """
    pass


def apply_dCRF(mode,
               src=None,
               save_path=None,
               save_path_logit=None,
               dataset=None,
               user="Ours",
               skip_exist_file=False):
    """
    Apply dense CRF in deeplab prediction or random walk prediction, then evaluate the dCRF result
    ---
    mode: rw | deeplab
    src: the path you want to perform dense CRF
    """
    t_start = time.time()
    """Code Zone"""
    # ===predction + CRF and save the result in args.path4saveCRF and evaluate meanIoU
    # >>specify your >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # descript = "Deeplabv2@ImageNet Pretrain Weight @ w/o CRF mIOU=58.26 @ evaluate on val set@CRF parameter: CRF_psa@"
    descript = "Our Complete pseudo label"
    if save_path is None:
        save_path = "submitVOC/{}/results/VOC2012/Segmentation/comp6_{}_cls".format(
            user, dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if dataset is None:
        f_list = args.path4val_images
    else:
        f_list = os.path.join("..", "psa", "voc12", "{}.txt".format(dataset))

    args.path4saveCRF = os.path.join(*(args.path4save_img.split('/')[:-1] +
                                       [args.path4saveCRF]))
    print("args.path4saveCRF: ", args.path4saveCRF)
    print("f_list: ", f_list)
    # mode = 'deeplab'  # rw | deeplab
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    path4saveCRF_rw = os.path.join("RW_CRF")
    path4saveCRF_deeplab = save_path
    print("savePath:", save_path)
    if mode == 'deeplab':
        evaluate_folder = path4saveCRF_deeplab
        if src is None:
            pred_root = os.path.join(
                args.path4Deeplab_logit,
                get_least_modify_file(args.path4Deeplab_logit))
        else:
            pred_root = os.path.join(args.path4Deeplab_logit, src)
        print("pred_root: ", pred_root)
    elif mode == 'rw':
        f_list = args.path4train_aug_images
        pred_root = os.path.join("/home", getpass.getuser(), "psa",
                                 "RES_RW_np")
        evaluate_folder = path4saveCRF_rw
    img_list = load_img_name_list(f_list)
    ignore_list = [os.path.splitext(f)[0] for f in os.listdir(save_path)]
    for idx, img in enumerate(img_list):
        if skip_exist_file:
            if img in ignore_list:
                print("[{}] ignore: {}".format(idx, img), end='\r')
                continue
        print("[{}/{}]========time: {} ".format(
            idx, np.size(img_list),
            show_timing(time_start=t_start, time_end=time.time())),
              end='\r')
        if mode == 'rw':
            crf_psa(img_name=img,
                    prediction_root=pred_root,
                    save_path=path4saveCRF_rw)  # use by psa, infer_cls.py
        elif mode == 'deeplab':
            crf_deepLab(img_name=img,
                        prediction_root=pred_root,
                        save_path=path4saveCRF_deeplab,
                        save_path_logit=save_path_logit,
                        CRF_parameter=args.CRF_deeplab)
    if dataset != "test":
        evaluate_dataset_IoU(file_list=f_list,
                             predicted_folder=evaluate_folder,
                             descript=descript + os.path.basename(pred_root),
                             path4GT=args.path4VOC_class_aug)


def mixPPLAndCPL(save_path=None, PPL_path=None, CPL_path=None):
    """
    Replace the prediction in train index of complete label with partial label (assume that GCN can not denoise)
    ---
    - PPL_path: default will take the least result
    - CPL_path: default will take the least result
    """
    from os.path import join as opj
    t_start = time.time()
    CPL_path = os.path.join(
        args.path4Complete_label_label,
        get_least_modify_file(args.path4Complete_label_label))
    PPL_path = os.path.join(
        args.path4partial_label_label,
        "RES_CAM_TRAIN_AUG_PARTIAL_PSEUDO_LABEL@PIL_near@confident_ratio_0.3_UP"
    )
    f_list = load_img_name_list(args.path4train_aug_images)
    length = len(f_list)
    additional_space = os.path.join("..", "..", "..", "media",
                                    getpass.getuser(), "pygcn", "data")
    if save_path is None:
        save_path = opj(additional_space,
                        "MixPartialPseudoLabelAndComPletePseudoLabel")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    for idx, label_name in enumerate(f_list, start=1):
        print("[{}/{}] Time:{}".format(
            idx, length, show_timing(time_start=t_start,
                                     time_end=time.time())),
              end='\r')
        cpl = opj(CPL_path, label_name + '.png')
        ppl = opj(PPL_path, label_name + '.png')
        cpl_arr = np.array(Image.open(cpl))
        ppl_arr = np.array(Image.open(ppl))
        final = np.where(ppl_arr == 255, cpl_arr, ppl_arr)
        # print("cpl_arr.shape ", cpl_arr.shape)
        # print("ppl_arr.shape ", ppl_arr.shape)
        # print("final.shape ", final.shape)
        scipy.misc.toimage(final, cmin=0, cmax=255, pal=colors_map,
                           mode="P").save(
                               os.path.join(save_path, label_name + '.png'))
    # === evaluation ===
    evaluate_dataset_IoU(
        file_list=args.path4train_aug_images,
        predicted_folder=save_path,
        path4GT=args.path4VOC_class_aug,
        descript="OurCPLmIoU=59.77@OurPPLmIoU=77.05@train_aug@afterCRF")


""" 2020.6.25 """


def get_least_modify_file(folder, show_sorted_data=False):
    dir_list = os.listdir(folder)
    dir_list = sorted(dir_list,
                      key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    if show_sorted_data:
        for it in dir_list:
            print("dir_list: {:<80}  time: {:}".format(
                it, os.path.getmtime(os.path.join(args.path4GCN_logit, it))))
    return dir_list[-1]


if __name__ == "__main__":
    fire.Fire()