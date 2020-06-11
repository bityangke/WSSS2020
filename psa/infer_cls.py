import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
from infer_aff import colors_map
import time


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="res38_cls.pth", type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--infer_list",
                        default="voc12/train_aug.txt",
                        type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default="VOC2012", type=str)
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument(
        "--out_cam",
        default=
        "RES_CAM_TRAIN_AUG",
        type=str)
    # parser.add_argument("--out_la_crf", default="RES_CAM_LA_TRAIN", type=str)
    parser.add_argument("--out_la_crf", default=None, type=str)
    # parser.add_argument("--out_la_crf", default=None, type=str)
    # parser.add_argument("--out_ha_crf", default="RES_CAM_HA_VAL", type=str)
    parser.add_argument("--out_ha_crf", default=None, type=str)
    parser.add_argument(
        "--out_cam_pred",
        default=
        "RES_CAM_LABEL_TRAIN_AUG",
        type=str)

    args = parser.parse_args()
    if not os.path.exists(args.out_cam):
        os.makedirs(args.out_cam)
    if not os.path.exists(args.out_cam_pred):
        os.mkdir(args.out_cam_pred)

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(
        args.infer_list,
        voc12_root=args.voc12_root,
        scales=(1, 0.5, 1.5, 2.0),
        inter_transform=torchvision.transforms.Compose(
            [np.asarray, model.normalize, imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    num_image = len(infer_data_loader)
    t_start = time.time()
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader, 1):
        img_name = img_name[0]
        label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    cam = model_replicas[i % n_gpus].forward_cam(img.cuda())
                    cam = F.upsample(cam,
                                     orig_img_size,
                                     mode='bilinear',
                                     align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1,
                                                                 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work,
                                            list(enumerate(img_list)),
                                            batch_size=12,
                                            prefetch_size=0,
                                            processes=args.num_workers)

        cam_list = thread_pool.pop_results()  # [len of img_list,21,H,W]

        sum_cam = np.sum(cam_list, axis=0)  # [21,H,W]
        # === devide each CAM value by the max value in each class CAM
        norm_cam = sum_cam / (np.max(sum_cam,
                                     (1, 2), keepdims=True) + 1e-5)  # [21,H,W]

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:  # label[i] is real number
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0]) * 0.2]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            # scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))
            scipy.misc.toimage(arr=pred.astype(np.uint8),
                               cmin=0,
                               cmax=255,
                               pal=colors_map,
                               mode="P").save(
                                   os.path.join(args.out_cam_pred,
                                                img_name + '.png'))

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            print("v.shape ", v.shape)
            print("orig_img.shape ", orig_img.shape)
            print("type(bgcam_score)", type(bgcam_score))
            print("type(orig_img) ", type(orig_img))
            crf_score = imutils.crf_inference(orig_img,
                                              bgcam_score,
                                              labels=bgcam_score.shape[0])
            print("after CRF......")
            input("[130].......")
            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key + 1] = crf_score[i + 1]

            return n_crf_al

        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(cam_dict, args.low_alpha)
            np.save(os.path.join(args.out_la_crf, img_name + '.npy'), crf_la)

        if args.out_ha_crf is not None:
            crf_ha = _crf_with_alpha(cam_dict, args.high_alpha)
            np.save(os.path.join(args.out_ha_crf, img_name + '.npy'), crf_ha)

        print("[{}/{}]: {}  time: {}".format(iter, num_image, img_name),
              show_timing(time_start=t_start, time_end=time.time()))
