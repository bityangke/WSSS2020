import numpy as np
import scipy.sparse as sp
import torch

import torch.nn.functional as F
from torch import nn
import scipy.misc
import os
from config import opt as args
from PIL import Image
import datetime
import fire
import time
import getpass
from cv2 import imread, imwrite
import pydensecrf.densecrf as dcrf
from utils import colors_map, load_img_name_list, show_timing
from utils import evaluate_dataset_IoU, load_image_label_from_xml
from utils import get_least_modify_file


def crf_inference(img, probs, CRF_parameter, scale_factor=1, labels=21):
    """
    this setting is different from PSA

    IoU=62.44
    """
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)
    pred_softmax = torch.nn.Softmax(dim=0)
    probs = pred_softmax(torch.tensor(probs)).numpy()
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=CRF_parameter["pos_xy_std"] / scale_factor,
                          compat=CRF_parameter["pos_w"])
    d.addPairwiseBilateral(sxy=CRF_parameter["bi_xy_std"] / scale_factor,
                           srgb=CRF_parameter["bi_rgb_std"],
                           rgbim=np.copy(img),
                           compat=CRF_parameter["bi_w"])
    Q = d.inference(CRF_parameter["iter_max"])
    return np.array(Q).reshape((n_labels, h, w))


def crf(img_name,
        CRF_parameter,
        save_path_label,
        save_path_logit,
        img=None,
        probs=None,
        prediction_root=None,
        scale_factor=1,
        labels=21):
    """
    CRF for prediction in `prediction_root`
    ---
    - 3 setting was provided
    - CRF_parameter: args.CRF | args.CRF_deeplab | args.psa
    - 
    """

    if img is None:
        img = imread(os.path.join(args.path4Image,
                                  img_name + '.jpg'))  # [H,W,3]
    H, W = img.shape[:2]
    # === load predict_dict ===
    if prediction_root is None:
        prediction_root = os.path.join("predict_result_matrix_visual_new",
                                       "250")
    prect_dict = np.load(os.path.join(prediction_root, img_name + '.npy'),
                         allow_pickle=True).item()
    """ ===3. save random walk prediction as np array in RW_prediction  === """
    def rw_crf(predicted_dict, name=None):
        """
        - orig_img: [H,W,3], np array
        - predicted_dict: dictionary, each item is a [H,W] predicted score for coresponding class
        """
        v = np.array(list(predicted_dict.values()))
        img_path = os.path.join(args.path4Image, name + '.jpg')
        orig_img = np.asarray(Image.open(img_path))
        # === note that orig_img have the shape [H,W,3]
        crf_score = crf_inference(orig_img,
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

    rw_crf_resut, crf_dict = rw_crf(predicted_dict=prect_dict, name=img_name)
    # === save as dictionary
    if not os.path.exists(save_path_logit):
        os.makedirs(save_path_logit)
    np.save(os.path.join(save_path_logit, img_name + '.npy'), crf_dict)
    """ ===4. save the prediction as label in save_path as .png === """
    if not os.path.exists(save_path_label):
        os.makedirs(save_path_label)
    scipy.misc.toimage(rw_crf_resut.argmax(axis=0),
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode="P").save(
                           os.path.join(save_path_label, img_name + '.png'))


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
    For Deeplab prediction
    ---
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
    prect_dict = dict()
    img_label = load_image_label_from_xml(img_name=img_name,
                                          voc12_root=args.path4VOC_root)
    prect_dict[0] = logits[0].numpy()
    # key range from 0~20 if you use VOC dataset
    for key, idx in enumerate(img_label):  # img_label +1 = segmentation_label
        if int(idx) > 0:
            prect_dict[key + 1] = logits[key + 1].numpy()
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
        crf_score = crf_inference(orig_img,
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


def help():
    print("To use CRF:")
    print("python CRF.py apply --[arg]=argValue")
    print("arg:'CRF_parameter', 'f_list', 'path4saveCRF', 'pred_root']")


def apply(**kwargs):
    """
    Apply dense CRF in pred_root (default: the least modify folder in args.path4GCN_label)
    ---
    parameter:
    - --CRF_parameter: args.CRF | args.CRF_deeplab | args.CRF_psa
    - --path4saveCRF_label: path for save CRF label
    - --path4saveCRF_logit: path for save CRF logit
    - --f_list: the .txt for VOC2012 dataset (e.g. train.txt)
    - --mode: other | rw | deeplab |  this funtion is still construct...
    """
    parameter_dict = dict()
    t_start = time.time()
    """Code Zone"""
    # ===predction + CRF and save the result in path4saveCRF and evaluate meanIoU
    # >>specify your >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_now = datetime.datetime.today()
    time_now = "{}_{}_{}_{}h{}m".format(time_now.year, time_now.month,
                                        time_now.day, time_now.hour,
                                        time_now.minute)
    descript = ""
    parameter_dict["CRF_parameter"] = args.CRF
    parameter_dict["path4saveCRF_label"] = os.path.join(
        args.path4Complete_label_label, time_now)
    parameter_dict["path4saveCRF_logit"] = os.path.join(
        args.path4Complete_label_logit, time_now)
    parameter_dict["pred_root"] = os.path.join(
        args.path4GCN_logit, get_least_modify_file(args.path4GCN_logit))
    data_len = len(os.listdir(parameter_dict["pred_root"]))
    if data_len == 10582:
        parameter_dict["f_list"] = args.path4train_aug_images
    elif data_len == 1464:
        parameter_dict["f_list"] = args.path4train_images
    elif data_len == 1449:
        parameter_dict["f_list"] = args.path4val_images
    elif data_len == 1456:
        parameter_dict["f_list"] = args.path4test_images

    parameter_dict["mode"] = "other"  # rw | deeplab
    evaluate_folder = parameter_dict["path4saveCRF_label"]
    img_list = load_img_name_list(parameter_dict["f_list"])
    # === load parameter
    for k, v in kwargs.items():
        if k in parameter_dict.keys():
            parameter_dict[k] = eval(v)
            print("{}: {}".format(k, parameter_dict[k]))
        else:
            print("There is no parameter: {}".format(k))
            print("use `python CRF.py help` to know how to use.")
            return

    print("path4saveCRF_label: ", parameter_dict["path4saveCRF_label"])
    print("pred_root: ", parameter_dict["pred_root"])
    for idx, img in enumerate(img_list):
        print("[{}/{}]  img: {} ==== time: {} ".format(
            idx, np.size(img_list), img,
            show_timing(time_start=t_start, time_end=time.time())),
              end='\r')
        crf(img_name=img,
            prediction_root=parameter_dict["pred_root"],
            save_path_label=parameter_dict["path4saveCRF_label"],
            save_path_logit=parameter_dict["path4saveCRF_logit"],
            CRF_parameter=parameter_dict["CRF_parameter"]
            )  # use by psa, infer_cls.py

    evaluate_dataset_IoU(file_list=parameter_dict["f_list"],
                         predicted_folder=evaluate_folder,
                         descript=descript,
                         path4GT=args.path4VOC_class_aug)
    """Code Zone"""
    show_timing(time_start=t_start, time_end=time.time())


if __name__ == "__main__":
    fire.Fire()
