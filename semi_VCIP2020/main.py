# from model import DeepLabV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tensorboardX import SummaryWriter
import time
import cv2
from PIL import Image
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from datasets import VOCDataSet
import matplotlib.pyplot as plt
import random
import numpy as np
# import argparse
import yaml
from addict import Dict
from libs.utils import DenseCRF, PolynomialLR, scores
from libs.models import *
from torchnet.meter import MovingAverageValueMeter
import fire
import datetime

colors_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [0, 0, 255]]
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434),
                    dtype=np.float32)
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

# BATCH_SIZE = 10
# DATA_DIRECTORY = '../psa/VOC2012'
# # DATA_LIST_PATH = './dataset/list/train_aug.txt'
# DATA_LIST_PATH = '../psa/voc12/train.txt'
# VAL_DATA_LIST_PATH = '../psa/voc12/val.txt'
# IGNORE_LABEL = 255
# INPUT_SIZE = '321,321'
# LEARNING_RATE = 2.5e-4
# MOMENTUM = 0.9
# NUM_CLASSES = 21
# NUM_STEPS = 20000
# POWER = 0.9
# RANDOM_SEED = 1234
# RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_COCO_init.pth'
# SAVE_NUM_IMAGES = 2
# SAVE_PRED_EVERY = 1000
# SNAPSHOT_DIR = './snapshots/'
# EVALUATE_EVERY = 200
# WEIGHT_DECAY = 0.0005
# TRAIN_LABEL = "SegmentationClass"  # "RES_RW"
# GPU_IDX = [0, 1, 2, 3, 4, 5, 6, 7]
CONFIG_PATH = 'configs/voc12.yaml'

ya_config = yaml.load(open(CONFIG_PATH), Loader=yaml.FullLoader)
CONFIG = Dict(ya_config, Loader=yaml.FullLoader)


def load_image_label_from_xml(img_name, voc12_root):
    """
    No background index be consider
    ===
    - img_name: e.g. 2007_000032
    - return np array lenth=20
    """
    from xml.dom import minidom

    el_list = minidom.parse(
        os.path.join(voc12_root, "Annotations",
                     img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in SEG_LIST:
            cat_num = CLS_NAME_TO_ID[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=CONFIG.DATASET.IGNORE_LABEL).cuda()
    return criterion(pred, label)


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from deeplab.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    # aveJ, j_list, M = ConfM.jaccard()
    return ConfM.jaccard()


def evaluate_gpu(model, writer, train_iter, save_class=False):
    """
    Create the model and start the evaluation process.
    ---
    - return meanIoU, class_IoU[ ]
    """
    torch.set_grad_enabled(False)
    device = get_device(torch.cuda.is_available())
    model.eval()

    testloader = data.DataLoader(VOCDataSet(CONFIG.DATASET.DIRECTORY,
                                            CONFIG.DATASET.VAL_LIST_PATH,
                                            crop_size=(CONFIG.IMAGE.SIZE.TEST,
                                                       CONFIG.IMAGE.SIZE.TEST),
                                            mean=IMG_MEAN,
                                            scale=False,
                                            mirror=False),
                                 batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
                                 shuffle=False,
                                 pin_memory=True)

    interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True)
    data_list = []
    t_start = time.time()
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, size, name = batch
            size = size[0]
            output = model(image.to(device))

            output = interp(output).squeeze(dim=0)  # [21,505,505]
            output = output[:, :size[0], :size[1]]
            gt = torch.tensor(label.detach().clone()[0][:size[0], :size[1]],
                              dtype=torch.int)  # [366,500]

            output = output.permute(1, 2, 0)  # [H,W,21]
            output = torch.tensor(output.clone().detach().argmax(dim=2),
                                  dtype=torch.int)
            print("Evaluate [{}]name {} Time:{:>4.1f} s".format(
                index, name,
                time.time() - t_start),
                  end='\r')

            data_list.append(
                [gt.flatten().cpu().numpy(),
                 output.flatten().cpu().numpy()])

    show_timing(t_start, time.time())
    aveJ, j_list, M = get_iou(data_list, CONFIG.DATASET.N_CLASSES)
    print("MeanIoU: {:2.2f}".format(aveJ * 100))
    writer.add_scalar("meanIoU", aveJ * 100, global_step=train_iter)
    return aveJ, j_list


def evaluate(model, writer, train_iter, save_class=False, save_logit=False):
    """Create the model and start the evaluation process."""
    # Configuration
    torch.set_grad_enabled(False)
    device = get_device(torch.cuda.is_available())
    model.eval()

    # Path to save logits
    logit_dir = os.path.join(
        CONFIG.EXP.SAVE_PRED,
        "logit",
    )
    if not os.path.exists(logit_dir):
        os.makedirs(logit_dir)
    print("Logit dst:", logit_dir)
    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.SAVE_PRED,
        "scores",
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores_xavier.json")
    print("Score dst:", save_path)

    # saved_state_dict = torch.load(args.test_weight_path)
    # model.load_state_dict(saved_state_dict)

    testloader = data.DataLoader(VOCDataSet(CONFIG.DATASET.DIRECTORY,
                                            CONFIG.DATASET.VAL_LIST_PATH,
                                            crop_size=(CONFIG.IMAGE.SIZE.TEST,
                                                       CONFIG.IMAGE.SIZE.TEST),
                                            mean=IMG_MEAN,
                                            scale=False,
                                            mirror=False),
                                 batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
                                 shuffle=False,
                                 pin_memory=True)

    t_start = time.time()
    # with torch.no_grad():
    preds, gts = [], []
    for index, batch in enumerate(testloader):
        images, gt_labels, size, image_names = batch
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)
        # Save on disk for CRF post-processing
        if save_logit:
            for image_id, logit in zip(image_names, logits):
                filename = os.path.join(logit_dir, image_id + ".npy")
                np.save(filename, logit.cpu().numpy())

        # Pixel-wise labeling
        _, H, W = gt_labels.shape
        logits = F.interpolate(logits,
                               size=(H, W),
                               mode="bilinear",
                               align_corners=False)
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)

        preds += list(labels.cpu().numpy())
        gts += list(gt_labels.numpy())
        score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)
        """ scores()
         return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
        }
        """
        print("Evaluate [{}] name {}  time: {:3} s".format(
            index, image_names[0],
            time.time() - t_start),
              end='\r')
        # === save predict result in .png
        if save_class:
            save_dir_class = os.path.join(
                CONFIG.EXP.SAVE_PRED,
                "classes",
            )
            if not os.path.exists(save_dir_class):
                os.makedirs(save_dir_class)
            scipy.misc.toimage(labels.cpu().numpy()[0],
                               cmin=0,
                               cmax=255,
                               pal=colors_map,
                               mode='P').save(
                                   os.path.join(save_dir_class,
                                                image_names[0] + '.png'))
        # show_timing(t_start, time.time())

    writer.add_scalar("meanIoU", score["Mean IoU"], train_iter)


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


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter)**(power))


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    ---
    - Other nearest methods result in misaligned labels.
    - -> F.interpolate(labels, shape, mode='nearest')
    - -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


def test(cuda=torch.cuda.is_available(), model_iteration=None,
         save_class=True):
    """
    Evaluation on validation set
    """
    time_now = datetime.datetime.today()
    time_now = "{}_{}_{}_{}h{}m".format(time_now.year, time_now.month,
                                        time_now.day, time_now.hour,
                                        time_now.minute)
    # Configuration
    device = get_device(cuda)
    torch.set_grad_enabled(False)
    model_path = os.path.join(CONFIG.MODEL.SAVE_PATH,
                              "VOC12_{}.pth".format(model_iteration))
    # Path to save logits
    logit_dir = os.path.join(
        CONFIG.EXP.SAVE_PRED,
        "logit" + "_" + time_now,
    )
    if not os.path.exists(logit_dir):
        os.makedirs(logit_dir)
    print("Logit dst:", logit_dir)
    # === save predict result in .png
    if save_class:
        save_dir_class = os.path.join(
            CONFIG.EXP.SAVE_PRED,
            "classes" + "_" + time_now,
        )
        if not os.path.exists(save_dir_class):
            os.makedirs(save_dir_class)
    print("classes dst:", save_dir_class)

    # DataLoader
    testloader = data.DataLoader(VOCDataSet(CONFIG.DATASET.DIRECTORY,
                                            CONFIG.DATASET.VAL_LIST_PATH,
                                            crop_size=(CONFIG.IMAGE.SIZE.TEST,
                                                       CONFIG.IMAGE.SIZE.TEST),
                                            mean=IMG_MEAN,
                                            scale=False,
                                            mirror=False),
                                 batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
                                 shuffle=False,
                                 pin_memory=True)

    # Model
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    # for the case that model weight name like "base.aspp.c1.weight"
    if model_path == "VOC12_7499.pth":

        def load_weight(state_dict, model):
            for k, v in state_dict.items():
                # print("key:{}".format(k))
                new_k = k[7:]
                # print("new_key:{}".format(new_k))
                # print("==================")
                model.load_state_dict({new_k: v}, strict=False)

        return model
        print("there is 'base' in state_dict")
        load_weight(state_dict, model)  # use for Our 60.10
    else:
        model.load_state_dict(state_dict)

    model = nn.DataParallel(model)
    model.eval()
    model.to(device)
    # Iterate through validation dataset
    interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True)
    data_list = []
    t_start = time.time()
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, size, name = batch
            size = size[0]
            # output = model(Variable(image, volatile=True).cuda(gpu0))
            output = model(image.to(device))  # [1,21,H_dn,W_dn]

            # Save on disk for CRF post-processing
            filename = os.path.join(logit_dir, name[0] + ".npy")
            np.save(filename, output.cpu().numpy())

            logits = interp(output).squeeze(dim=0)  # [21,505,505]
            logits = logits[:, :size[0], :size[1]]  # === [21,H,W]
            gt = torch.tensor(label.clone().detach()[0][:size[0], :size[1]],
                              dtype=torch.int)  # === [H,W]

            # ==== save in dictionary===
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
                img_label = load_image_label_from_xml(
                    img_name=img_name, voc12_root=CONFIG.DATASET.DIRECTORY)
                pseudo_label_dict[0] = seg_score[0]
                # key range from 0~20 if you use VOC dataset
                for key in img_label:  # img_label +1 = segmentation_label
                    pseudo_label_dict[int(key + 1)] = seg_score[int(key + 1)]
                # save score
                if save_npy:
                    np.save(os.path.join(destination_np, img_name),
                            pseudo_label_dict)

            # ==========================
            logits = logits.permute(1, 2, 0)  # [H,W,21]
            pred = torch.tensor(logits.clone().detach().argmax(dim=2),
                                dtype=torch.int)
            print("Evaluate [{}]name {} Time:{:>4.1f} s".format(
                index, name[0],
                time.time() - t_start),
                  end='\r')

            data_list.append(
                [gt.flatten().cpu().numpy(),
                 pred.flatten().cpu().numpy()])

            # Pixel-wise labeling
            scipy.misc.toimage(pred.cpu().numpy(),
                               cmin=0,
                               cmax=255,
                               pal=colors_map,
                               mode='P').save(
                                   os.path.join(save_dir_class,
                                                name[0] + '.png'))

    aveJ, j_list, M = get_iou(data_list, CONFIG.DATASET.N_CLASSES)
    print(show_timing(t_start, time.time()))
    print("=" * 34)
    for idx, iu_class in enumerate(j_list):
        print("{:12}: {:>17.2f} %".format(SEG_ID_TO_NAME[idx], iu_class * 100))
    print("=" * 34)

    print("MeanIoU: {:2.2f}".format(aveJ * 100))

    # score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)
    # """ scores()
    # return {
    #     "Pixel Accuracy": acc,
    #     "Mean Accuracy": acc_cls,
    #     "Frequency Weighted IoU": fwavacc,
    #     "Mean IoU": mean_iu,
    #     "Class IoU": cls_iu,
    # }
    # """


def train():
    """Create the model and start the training."""
    # === 1.Configuration
    print(CONFIG_PATH)
    # === select which GPU you want to use
    # === here assume to use 8 GPUs, idx are 0,1,2,3,...,7
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, CONFIG.EXP.GPU_IDX))

    device = get_device(torch.cuda.is_available())
    cudnn.benchmark = True
    comment_init = ""
    writer = SummaryWriter(comment=comment_init)  # Setup loss logger
    # === MovingAverageValueMeter(self,windowsize)
    # === - add(value)： 记录value
    # === - reset()
    # === - value() ： 返回MA和标准差
    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)
    if not os.path.exists(CONFIG.MODEL.SAVE_PATH):
        os.makedirs(CONFIG.MODEL.SAVE_PATH)
    # Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,  # ./data
        "models",
        CONFIG.MODEL.NAME.lower(),  # DeepLabV2_ResNet101_MSC
        CONFIG.DATASET.SPLIT.TRAIN,  # train_aug
    )
    # === checkpoint_dir: ./data/DeepLabV2_ResNet101_MSC/train_aug
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    # === 2.Dataloader ===
    trainloader = data.DataLoader(
        VOCDataSet(
            CONFIG.DATASET.DIRECTORY,
            CONFIG.DATASET.LIST_PATH,
            max_iters=CONFIG.SOLVER.ITER_MAX * CONFIG.SOLVER.BATCH_SIZE.TRAIN,
            crop_size=(CONFIG.IMAGE.SIZE.TRAIN, CONFIG.IMAGE.SIZE.TRAIN),
            scale=CONFIG.DATASET.RANDOM.SCALE,
            mirror=CONFIG.DATASET.RANDOM.MIRROR,
            mean=IMG_MEAN,
            label_path=CONFIG.DATASET.SEG_LABEL),  # for training 
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        shuffle=True,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        pin_memory=True)

    # 使用iter(dataloader)返回的是一个迭代器，可以使用next访问
    # loader_iter = iter(trainloader)

    # === 3.Create network & weights ===
    print("Model:", CONFIG.MODEL.NAME)

    # model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    model = DeepLabV2_DRN105_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL)
    # model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    print("    Init:", CONFIG.MODEL.INIT_MODEL)
    # === show the skip weight
    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)

    # === DeepLabv2 = Res101+ASPP
    # === model.base = DeepLabv2
    # === model = MSC(DeepLabv2)
    # model.base.load_state_dict(state_dict,
    #                            strict=False)  # strict=False to skip ASPP
    model = nn.DataParallel(model)  # multi-GPU
    model.to(device)  # put in GPU is available
    # === 4.Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)  # put in GPU is available

    # === 5.optimizer ===
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )
    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    time_start = time.time()  # set start time
    # === training iteration ===
    for i_iter, batch in enumerate(trainloader, start=1):
        torch.set_grad_enabled(True)
        model.train()
        model.module.base.freeze_bn()
        optimizer.zero_grad()
        images, labels, _, _ = batch

        logits = model(images.to(device))
        # <<<<<<<<<<<<<<<<<<<<
        # === Loss
        # === logits = [logits] + logits_pyramid + [logits_max]
        iter_loss = 0
        loss = 0
        for logit in logits:
            # Resize labels for {100%, 75%, 50%, Max} logits
            _, _, H, W = logit.shape
            labels_ = resize_labels(labels, size=(H, W))
            iter_loss += criterion(logit, labels_.to(device))
        # iter_loss /= CONFIG.SOLVER.ITER_SIZE
        iter_loss /= 4
        iter_loss.backward()
        loss += float(iter_loss)

        average_loss.add(loss)
        # Update weights with accumulated gradients
        optimizer.step()

        # Update learning rate
        scheduler.step(epoch=i_iter)

        # TensorBoard
        writer.add_scalar("loss", average_loss.value()[0], global_step=i_iter)
        print(
            'iter/max_iter = [{}/{}]  completed, loss = {:4.3} time:{}'.format(
                i_iter, CONFIG.SOLVER.ITER_MAX,
                average_loss.value()[0], show_timing(time_start, time.time())))
        # print('iter = ', i_iter, 'of', args.num_steps, '',
        #       loss.data.cpu().numpy())

        # === save final model
        if i_iter >= CONFIG.SOLVER.ITER_MAX:
            print('save final model as...{}'.format(
                osp.join(CONFIG.MODEL.SAVE_PATH,
                         'VOC12_' + str(CONFIG.SOLVER.ITER_MAX) + '.pth')))
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.MODEL.SAVE_PATH,
                         'VOC12_' + str(CONFIG.SOLVER.ITER_MAX) + '.pth'))
            break
        if i_iter % CONFIG.EXP.EVALUATE_ITER == 0:
            print("Evaluation....")
            evaluate_gpu(model, writer, i_iter)

        # === Save model every 250 iteration==========================
        # because DataParalel will add 'module' in each name of layer.
        # so here use model.module.state_dict()
        # ============================================================
        if i_iter % CONFIG.MODEL.SAVE_EVERY_ITER == 0:
            print('saving model ...')
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.MODEL.SAVE_PATH,
                         'VOC12_{}.pth'.format(i_iter)))


if __name__ == "__main__":
    fire.Fire()
    # train()
    # test(cuda=torch.cuda.is_available(), model_iteration=500)

    # ===============================================================
    # model = DeepLabV2(n_classes=21,
    #                   n_blocks=[3, 4, 23, 3],
    #                   atrous_rates=[6, 12, 18, 24])

    # checkpoint = torch.load('deeplabv2_resnet101_msc-vocaug-20000.pth')
    # model.load_state_dict(checkpoint, strict=False)
    # model.append_ASPP()
    # # model = model.cuda()
    # model.eval()

    # criterion = nn.CrossEntropyLoss()

    # ### input for DeepLabv2
    # image = torch.randn(1, 3, 513, 513)

    # ### target
    # target = torch.ones(1, 65, 65).long()

    # ### training input/output/loss
    '''
    output = model(image).view(-1,21,65*65)
    target = target.view(-1,65*65)
    loss = criterion(output, target)
    print(loss)
    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
    '''
    # ===============================================================
