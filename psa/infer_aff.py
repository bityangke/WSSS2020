import torch
import torchvision
from tool import imutils

import argparse
import importlib
import numpy as np

from PIL import Image
import voc12.data
from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path
import pickle
colors_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [0, 0, 255]]

SEG_LIST = [
    'BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
# transfer segmentation_class_name <-> segmentation_class_ID
SEG_NAME_TO_ID = dict(zip(SEG_LIST, range(len(SEG_LIST))))
SEG_ID_TO_NAME = dict(zip(np.arange(len(SEG_LIST)), SEG_LIST))
IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
# transfer image_class_name <-> image__class_ID
CLS_NAME_TO_ID = dict(zip(SEG_LIST[1:], range(len(SEG_LIST[1:]))))
CLS_ID_TO_NAME = dict(zip(np.arange(len(SEG_LIST[1:])), SEG_LIST[1:]))


def get_indices_in_radius(height, width, radius):

    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius - 1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(
        full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:

        indices_to = full_indices[dy:dy + cropped_height, radius_floor +
                                  dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to


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
    time_hms = "Total time elapsed: {:.0f} h {:.0f} m {:.0f} s\n".format(
        (time_end - time_start) // 3600, (time_end - time_start) / 60 % 60,
        (time_end - time_start) % 60)
    if show:
        print(time_hms)
    return time_hms


def get_parse():
    """  
    - there are 4 arguments you can change:
        - save_RW_predictionas_label: save random walk prediction as label in `.png`
        - save_aff_mat: save affinity matrix as np array in `.npy`
        - save_aff_mat_nomalize: save normalized affinity matrix as np array in `.npy`
        - save_RW_predictionas_np: save random walk predicted distribution as np array in `.npy`
    
    - this code will save the node feature automatically in ./AFF_FEATURE
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", required=True, type=str)
    # parser.add_argument("--network", default="network.vgg16_aff", type=str)
    # parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    # parser.add_argument("--num_workers", default=8, type=int)
    # parser.add_argument("--cam_dir", required=True, type=str)
    # parser.add_argument("--voc12_root", required=True, type=str)

    parser.add_argument("--weights", default="res38_aff.pth",
                        type=str)  # vgg_gn_aff | res38_aff
    parser.add_argument("--network", default="network.resnet38_aff",
                        type=str)  # network.vgg16_aff | network.resnet38_aff
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default="VOC2012", type=str)
    parser.add_argument("--infer_list",
                        default="voc12/train_aug.txt",
                        type=str)
    parser.add_argument("--cam_dir", default="RES_CAM_TRAIN_AUG",
                        type=str)  # VGG_CAM_TRAIN | RES_CAM_TRAIN
    parser.add_argument("--out_rw", default="RES_RW",
                        type=str)  # VGG_RW | RES_RW
    parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument("--num-class", default=21, type=int)
    parser.add_argument("--beta", default=8, type=int)
    parser.add_argument("--logt", default=8, type=int)
    parser.add_argument("--save_RW_predictionas_label",
                        default=False,
                        type=bool)
    parser.add_argument("--save_aff_mat_nomalize", default=True, type=bool)
    # === usuallly use these two
    parser.add_argument("--save_RW_predictionas_np", default=False, type=bool)
    parser.add_argument("--save_aff_mat", default=False, type=bool)
    parser.add_argument("--out_rw_crf", default="RW_CRF_label",
                        type=str)  # VGG_RW | RES_RW

    return parser.parse_args()


if __name__ == '__main__':
    # dataset = args.infer_list.split("/")[-1].split(".")[0]
    args = get_parse()

    model = getattr(importlib.import_module(args.network), 'Net')()

    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ImageDataset(
        args.infer_list,
        voc12_root=args.voc12_root,
        transform=torchvision.transforms.Compose(
            [np.asarray, model.normalize, imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    import time
    t_start = time.time()
    num_data = len(infer_data_loader)
    """ === do RW for each image === """
    for iter, (name, img) in enumerate(infer_data_loader, 1):
        name = name[0]
        if args.save_RW_predictionas_np:
            print("[{}/{}] {} time:{} min".format(
                iter, num_data, name, (time.time() - t_start) // 60),
                  end='\r')
        else:
            print("[{}/{}] {}".format(iter, num_data, name), end='\r')

        orig_shape = img.shape
        # print("orig_shape: ", orig_shape)
        padded_size = (int(np.ceil(img.shape[2] / 8) * 8),
                       int(np.ceil(img.shape[3] / 8) * 8))

        p2d = (0, padded_size[1] - img.shape[3], 0,
               padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)  # 長寬為8的倍數,不足的補0 [3,H,W]
        dheight = int(np.ceil(img.shape[2] / 8))
        dwidth = int(np.ceil(img.shape[3] / 8))
        if not os.path.exists(args.cam_dir):
            os.makedirs(args.cam_dir)
        cam = np.load(os.path.join(args.cam_dir, name + '.npy'),
                      allow_pickle=True).item()

        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k + 1] = v
        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:],
                                      (0), keepdims=False))**args.alpha
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])),
                              mode='constant')
        # print("[212] img.shape ", img.shape)
        with torch.no_grad():
            aff_mat = torch.pow(
                model.forward(img.cuda(), True, name=name, save_feature=True),
                args.beta)

            # print("aff_mat.shape {}".format(aff_mat.shape))
            """1. === save aff_mat, you can use numpy.load() to read. remenber set `allow_pickle` = True === """
            if args.save_aff_mat:
                # with open(args.out_rw + '/' + name + '.pkl', 'wb') as handle:
                #     pickle.dump(aff_mat.cpu().numpy(), handle)
                """ ===1.5  save aff_mat in npy === """
                path4aff_map_np = "AFF_MAT_NP_2020_{}".format(
                    args.weights.split("_")[0])
                if not os.path.exists(path4aff_map_np):
                    os.mkdir(path4aff_map_np)
                np.save(os.path.join(path4aff_map_np, name),
                        aff_mat.cpu().numpy())
            """ === normalize === """
            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(args.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)
            """2. === save aff_map_normalize as np.array in ./aff_map_normalize === """
            path4aff_mat_normalize = os.path.join(
                "psa", "AFF_MAT_normalize")
            if args.save_aff_mat_nomalize:
                if not os.path.exists(path4aff_mat_normalize):
                    os.makedirs(path4aff_mat_normalize)
                np.save(os.path.join(path4aff_mat_normalize, name),
                        trans_mat.cpu().data.numpy())

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)
            cam_vec = cam_full_arr.view(21, -1)
            cam_rw = torch.matmul(
                cam_vec.cuda(), trans_mat
            )  # [21,H_dn*W_dn], [21,H_dn*W_dn]x[H_dn*W_dn,H_dn*W_dn]
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)  # [1,21,H_dn,W_dn]
            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]),
                                       mode='bilinear')(cam_rw)  # [1,21,H,W]
            """ ===3. save random walk prediction as np array in RW_prediction  === """
            path4RW_prediction_np = "RW_prediction_np_{}".format(
                args.weights.split("_")[0])
            if args.save_RW_predictionas_np:
                if not os.path.exists(path4RW_prediction_np):
                    os.mkdir(path4RW_prediction_np)

                # >>>> save in dictionary() >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                img_label = load_image_label_from_xml(
                    img_name=name, voc12_root=args.voc12_root)
                predict_dict = dict()
                predict_dict[0] = cam_rw.cpu().data.numpy()[0][  # tensor
                    0, :orig_shape[2], :orig_shape[3]]  # background

                for idx, cls_ in enumerate(img_label):
                    if int(cls_) > 0:
                        # print("key:{} ID:{}".format(idx + 1,
                        #                             SEG_ID_TO_NAME[idx + 1]))
                        predict_dict[idx + 1] = cam_rw.cpu().data.numpy(
                        )[0][idx + 1, :orig_shape[2], :orig_shape[3]]
                        # print("predict_dict[idx + 1].shape ",
                        #       predict_dict[idx + 1].shape)

                def rw_crf(predicted_dict, name=None):
                    """
                    - orig_img: [3,H,W], np array
                    - predicted_dict: dictionary, each item is a [H,W] predicted score for coresponding class
                    """
                    v = np.array(list(predicted_dict.values()))
                    pred_softmax = torch.nn.Softmax(dim=0)
                    probs = pred_softmax(torch.tensor(v)).numpy()
                    # orig_img = orig_img.data.cpu().numpy().transpose((1, 2, 0))  # [H,W,3]
                    img_path = voc12.data.get_img_path(name, args.voc12_root)
                    orig_img = np.asarray(Image.open(img_path))
                    # === note that orig_img have the shape [H,W,3]
                    crf_score = imutils.crf_inference(orig_img,
                                                      probs,
                                                      labels=v.shape[0])
                    h, w = orig_img.shape[:2]
                    crf_score_np = np.zeros(shape=(args.num_class, h, w))
                    crf_dict = dict()
                    for i, key in enumerate(predicted_dict.keys()):
                        crf_score_np[key] = crf_score[i]
                        crf_dict[key] = crf_score[i]
                    return crf_score_np, crf_dict

                # === apply CRF
                # === note that orig_img must be in shape [H,W,3]
                # rw_crf_resut, crf_dict = rw_crf(predicted_dict=predict_dict,
                #                                 name=name)
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # === save as dictionary
                if not os.path.exists("RW_np"):
                    os.mkdir("RW_np")
                np.save(os.path.join("RW_np", name + '.npy'), predict_dict)

                # np.save(os.path.join(path4RW_prediction_np, name + '.npy'),
                #         cam_rw.cpu().data[0][:orig_shape[2], :orig_shape[3]])
            """ ===4. save the random walk prediction as label in args.out_rw as .png === """
            if args.save_RW_predictionas_label:
                if not os.path.exists(args.out_rw):
                    os.makedirs(args.out_rw)
                _, cam_rw_pred = torch.max(cam_rw, 1)  # [1,H,W]
                res = np.uint8(
                    cam_rw_pred.cpu().data[0])[:orig_shape[2], :
                                               orig_shape[3]]  # [H,W]
                scipy.misc.toimage(res,
                                   cmin=0,
                                   cmax=255,
                                   pal=colors_map,
                                   mode="P").save(
                                       os.path.join(args.out_rw,
                                                    name + '.png'))

    show_timing(t_start, time.time())
