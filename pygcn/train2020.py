# coding=UTF-8
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from PIL import Image
import os
import scipy
import scipy.misc
import fire
from cv2 import imread, imwrite
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import opt as args
from models import GCN
from dataset import graph_voc
from tensorboardX import SummaryWriter
import getpass
from utils import IOUMetric, colors_map, evaluate_dataset_IoU
from utils import load_image_label_from_xml, CLS_NAME_TO_ID, CLS_ID_TO_NAME
from utils import ANNOT_FOLDER_NAME, SEG_NAME_TO_ID, SEG_ID_TO_NAME
from utils import crf_inference_inf, load_img_name_list, show_timing
from utils import HLoss, symmetricLoss
import datetime
from models import GAT, SpGAT

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def postprocess_image_save(epoch,
                           model_output,
                           img_name="2007_009788",
                           save_prediction_np=False,
                           save_CRF=True,
                           rgbxy_t=None):
    """
    1.upsampling prediction_score
    2.save prediction scores (option)
    3.save prediction mask
    """
    print("=== Post Processing...")

    # load image as nd_array
    img = imread(os.path.join(args.path4img, img_name + ".jpg"))
    H_original, W_original, C = img.shape
    H, W = int(np.ceil(H_original / args.output_rate)), int(
        np.ceil(W_original / args.output_rate))

    # === tansfer shape(H*W,num_class) -> shape(num_class,H,W)
    model_output = model_output.reshape(H, W,
                                        model_output.size()[-1]).permute(
                                            2, 0, 1)
    print("model_output.shape: {}".format(model_output.shape))
    # === use bilinear to upsample the predicted mask
    upsampling = torch.nn.Upsample(size=(H_original, W_original),
                                   mode='bilinear',
                                   align_corners=True)
    model_output = model_output.unsqueeze(dim=0)  # [1,C,H,W]
    up_predict = upsampling(model_output).squeeze(
        dim=0)  # [C,H_original,W_original]
    # >>>>>>>>>> label propagation
    if args.use_LP:
        # [H,W,3]
        img = np.array(
            Image.open(os.path.join(args.path4Image, img_name + '.jpg')))
        rgb = torch.Tensor(img / 255.)  # [H_original,W_original,3]
        rgb = rgb.reshape(H_original * W_original,
                          3)  # [H_original*W_original,3]
        W = gaussian_propagator(features=rgb)  # [H_ori*W_ori,H_ori*W_ori]
        input("W.shape {}".format(W.shape))
        identity = torch.eye(n=W.shape[0])
        D_inv = torch.pow(torch.diag(torch.sum(W, dim=1)).inverse(),
                          exponent=.5)
        S = D_inv.matmul(W.matmul(D_inv))
        # [C,H,W]-> [C,H*W] -> [H_ori*W_ori,C]
        pred_LP = torch.matmul(
            torch.inverse(identity.cuda() - args.alpha4LP * S),
            up_predict.reshape(-1, W.shape[0]).permute(1,
                                                       0))  # [H_ori*W_ori,21]
        pred_LP = pred_LP.argmax(dim=1)  # [H_ori*W_ori]
        pred_LP = pred_LP.reshape(H_original, W_original)  # [H_ori,W_ori]
        # ============= save LP =================
        if not os.path.isdir(args.path4save_LP):
            os.makedirs(args.path4save_LP)
        scipy.misc.toimage(pred_LP.cpu().numpy(),
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(args.path4save_LP,
                                            img_name + '.png'))

    # >>>>>>>>>>
    up_predict_mask = torch.argmax(up_predict, dim=0)
    # === save the prediction score in dictionary
    # === in os.path.join(args.path4GCN_logit, str(epoch) )
    if save_prediction_np:
        path = args.path4GCN_logit
        if not os.path.exists(path):
            os.makedirs(path)
            print("GCN prediction save path:", path)
        up_predict_np = torch.exp(up_predict.clone()).cpu().numpy()
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        predict_dict = dict()
        predict_dict[0] = up_predict_np[0]
        for idx, cls_ in enumerate(img_label):
            if int(cls_) > 0:
                print("key:{} ID:{}".format(idx + 1, SEG_ID_TO_NAME[idx + 1]))
                # === note that model prediction is log(p) -> p = exp{log(p)}
                predict_dict[idx + 1] = up_predict_np[idx + 1]
        np.save(os.path.join(path, img_name + ".npy"), predict_dict)

    # === save the prediction as label
    # === in os.path.join(path4save, img_name + '.png')
    path4save = args.path4GCN_label
    if not os.path.isdir(path4save):
        os.makedirs(path4save)
    scipy.misc.toimage(up_predict_mask.cpu().numpy(),
                       cmin=0,
                       cmax=255,
                       pal=colors_map,
                       mode='P').save(
                           os.path.join(path4save, img_name + '.png'))
    print("image:{} save in {}!\n".format(img_name, epoch))


def test(model,
         features,
         labels,
         adj,
         idx_train,
         idx_test,
         dataloader,
         img_name,
         epoch,
         t4epoch,
         propagator=None):
    """
    evaluate loss & acc & IoU
    save IoU & Accuracy in "evaluation4image.txt"
    """
    model.eval()
    output = model(features_t, adj).detach()
    predictions = torch.argmax(output, dim=1).cpu().numpy()
    mask_gt = Image.open(os.path.join(args.path4Class, img_name + '.png'))
    mask_gt = np.asarray(mask_gt)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    # === save prediction mask and scores
    IoU_one_image = IOUMetric(args.num_class)
    IoU_one_image.add_batch(predictions.cpu().numpy(), mask_gt)
    acc, acc_cls, iu, mean_iu_tensor, fwavacc = IoU_one_image.evaluate()
    # === show information
    print("[{:03d}]=== Information:\n".format(epoch + 1),
          'mean_IoU: {:>8.5f}'.format(mean_iu_tensor.item()),
          'acc: {:>11.5f}'.format(acc),
          'loss_train: {:<8.4f}'.format(loss_train.item()),
          'time: {:<8.4f}s'.format(time.time() - t4epoch))

    # === save Information
    print("save accuracy:" + img_name + ' predict')
    with open("evaluation4image.txt", 'a') as f:
        f.write(img_name + "\t")
        f.write("IoU" + str(mean_iu_tensor.item()) + "\t")
        f.write("Acc:" + str(acc) + "\n")


def normalize_rgbxy(rgbxy):
    """devide rgbxy by its standard deviation"""
    var = torch.var(rgbxy, dim=0, unbiased=True)
    return rgbxy / torch.sqrt(var)


def gaussian_propagator(features):
    """
    Calculus gaussian kernal propagator for label propagation 
    ---
    features: a [num_nodes, len_feature] tensor
    e.g. for 10 nodes RGBXY the features.shape=(10,5)
    return the tensor (cpu)
    """
    output = features.cpu()  # [n,f]
    num_nodes, len_feature = output.shape
    rgbxy_repeate = output.repeat(num_nodes, 1)  # [n,f]
    rgbxy_np = output.numpy()
    rgbxy = torch.from_numpy(rgbxy_np.repeat(num_nodes, 0))  # .cuda()
    rgbxy = rgbxy - rgbxy_repeate
    rgbxy = rgbxy.permute(1, 0)
    rgbxy = rgbxy.view(len_feature, num_nodes,
                       num_nodes)  # 5 is the dim of feature (i.e. rgbxy)
    # variance was already absorted to RGBXY, so we don't devide variance here
    S = torch.exp(-torch.pow(torch.norm(rgbxy, dim=0), 2) / 2.)
    return S.cuda()


def evaluate_IoU(model=None,
                 features=None,
                 rgbxy=None,
                 adj=None,
                 img_name='2007_000039',
                 epoch=0,
                 img_idx=0,
                 writer=None,
                 IoU=None,
                 IoU_CRF=None,
                 save_prediction_np=False,
                 use_CRF=True):
    """
    this func only calculus Iou for batch images
    for whole dataset, please refer to evaluate_dataset_IoU()
    mask_predict: numpy
    """
    model.eval()  # so dropout can perform on test mode
    predictions_score = model(features, adj).detach()

    # === upsampling & save predicted mask, scores
    postprocess_image_save(img_name=img_name,
                           model_output=predictions_score,
                           epoch=epoch,
                           save_prediction_np=save_prediction_np,
                           save_CRF=use_CRF)

    # === evaluate IoU & Acc
    # GT segmentation label
    mask_gt = Image.open(os.path.join(args.path4Class, img_name + '.png'))
    mask_gt = np.asarray(mask_gt)

    # === model prediction w/o CRF
    mask_predit = Image.open(
        os.path.join(args.path4GCN_label, img_name + '.png'))
    mask_predit = np.asarray(mask_predit)

    # === model prediction w/ LP
    if args.use_LP:
        mask_LP = Image.open(os.path.join(args.path4save_LP,
                                          img_name + '.png'))
        mask_LP = np.asarray(mask_LP)
        IoU_CRF.add_batch(mask_LP, mask_gt)
        accLP, acc_clsLP, iuLP, mean_iu_tensorLP, fwavaccLP = IoU_CRF.evaluate(
        )
        writer.add_scalar("IoU_LP",
                          mean_iu_tensorLP.cpu().numpy(),
                          global_step=img_idx)

    # ==== evaluation ======
    IoU.add_batch(mask_predit, mask_gt)
    acc, acc_cls, iu, mean_iu_tensor, fwavacc = IoU.evaluate()
    writer.add_scalar("IoU", mean_iu_tensor.cpu().numpy(), global_step=img_idx)
    print("Acc: {:>11.2f} IoU: {:>11.2f} %\n".format(
        acc,
        mean_iu_tensor.cpu().numpy() * 100))


def debug(message="", line=None):
    if line:
        print("[{}] ".format(line), end=' ')
    print(message)
    input("debug---->")


def trainGAT(**kwargs):
    t_start = time.time()
    # 根据命令行参数更新配置
    args.parse(**kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 把有改動的參數寫到tensorboard名稱上
    comment_init = ''
    for k, v in kwargs.items():
        comment_init += '|{} '.format(v)
    writer = SummaryWriter(comment=comment_init)

    # === set evaluate object for evaluate later
    IoU = IOUMetric(args.num_class)
    IoU_CRF = IOUMetric(args.num_class)

    # === dataset
    train_dataloader = graph_voc()

    for ii, data in enumerate(train_dataloader):
        if True:
            """ Still building................"""
            H, W, C = data["rgbxy_t"].shape
            A = torch.zeros([H * W, H * W], dtype=torch.float64)

            def find_neibor(card_x, card_y, H, W):
                """
                Return idx of neibors of (x,y) in list
                ---
                """
                neibors_idx = []
                for idx_x in [card_x - 1, card_x, card_x + 1]:
                    for idx_y in [card_y - 1, card_y, card_y + 1]:
                        if (-1 < idx_x < H) and (-1 < idx_y < W):
                            neibors_idx.append(
                                (idx_x * W + idx_y, idx_x, idx_y))
                return neibors_idx

            t_start = time.time()
            neibors = dict()
            for node_idx in range(H * W):
                card_x, card_y = node_idx // W, node_idx % W
                neibors = find_neibor(card_x, card_y, H, W)
                # print("H:{} W:{} | {} -> ({},{})".format(
                # H, W, node_idx, card_x, card_y))
                for nei in neibors:
                    # print("nei: ", nei)
                    diff = data["rgbxy_t"][
                        card_x, card_y, :3] - data["rgbxy_t"][nei[1],
                                                              nei[2], :3]
                    A[node_idx, nei[0]] = torch.exp(
                        -torch.pow(torch.norm(diff), 2) /
                        (2. * args.CRF_deeplab["bi_rgb_std"]))
            print("{:3.1f}s".format(time.time() - t_start))
            D = torch.diag(A.sum(dim=1))
            L_mat = D - A

        # === Model and optimizer
        img_label = load_image_label_from_xml(img_name=data["img_name"],
                                              voc12_root=args.path4VOC_root)
        img_class = [idx + 1 for idx, f in enumerate(img_label) if int(f) == 1]
        num_class = np.max(img_class) + 1
        # Model and optimizer
        sparse = True
        if sparse:
            model = SpGAT(nfeat=data["features_t"].shape[1],
                          nhid=args.hidden_GAT,
                          nclass=args.num_class,
                          dropout=args.dropout_GAT,
                          nheads=args.nb_heads_GAT,
                          alpha=args.alpha_GAT)
        else:
            model = GAT(nfeat=data["features_t"].shape[1],
                        nhid=args.hidden_GAT,
                        nclass=args.num_class,
                        dropout=args.dropout_GAT,
                        nheads=args.nb_heads_GAT,
                        alpha=args.alpha_GAT)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr_GAT,
                               weight_decay=args.weight_decay_GAT)

        # ==== moving tensor to GPU
        if args.cuda:
            model.to(device)
            data["features_t"] = data["features_t"].to(device)
            data["adj_t"] = data["adj_t"].to(device)
            data["labels_t"] = data["labels_t"].to(device)
            data["label_fg_t"] = data["label_fg_t"].to(device)
            data["label_bg_t"] = data["label_bg_t"].to(device)
            L_mat = L_mat.to(device)

        # === save the prediction before training
        if args.save_mask_before_train:
            model.eval()
            postprocess_image_save(img_name=data["img_name"],
                                   model_output=model(data["features_t"],
                                                      data["adj_t"]).detach(),
                                   epoch=0)

        # ==== Train model
        t4epoch = time.time()
        criterion_ent = HLoss()
        # criterion_sym = symmetricLoss()
        print("fg label: ", torch.unique(data["label_fg_t"]))
        print("bg label: ", torch.unique(data["label_bg_t"]))
        print("img_class: ", img_class)
        for epoch in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()
            output = model(data["features_t"], data["adj_t"])
            print("output.shape ", output.shape)
            print("output....")
            # === seperate FB/BG label
            loss_fg = F.nll_loss(output, data["label_fg_t"], ignore_index=255)
            loss_bg = F.nll_loss(output, data["label_bg_t"], ignore_index=255)
            loss_entmin = criterion_ent(output,
                                        data["labels_t"],
                                        ignore_index=255)
            # loss_sym = criterion_sym(output, labels_t, ignore_index=255)
            loss_lap = torch.trace(
                torch.mm(output.transpose(1, 0),
                         torch.mm(L_mat.type_as(output), output))) / (H * W)
            gramma = 1e-3
            loss_lap *= gramma
            # print("gramma*loss_lap:{}".format(loss_lap.data.item()))
            loss = loss_fg + loss_bg + loss_entmin + loss_lap

            if loss is None:
                print("skip this image: ", data["img_name"])
                break

            loss_train = loss.cuda()
            loss_train.backward()
            optimizer.step()
            print("{}/{}".format(epoch, args.max_epoch))
            # === save predcit mask of LP at max epoch & IoU of img
            if (epoch + 1) % args.max_epoch == 0 and args.save_mask:
                evaluate_IoU(model=model,
                             features=data["features_t"],
                             adj=data["adj_t"],
                             img_name=data["img_name"],
                             epoch=args.max_epoch,
                             img_idx=ii + 1,
                             writer=writer,
                             IoU=IoU,
                             IoU_CRF=IoU_CRF,
                             use_CRF=False,
                             save_prediction_np=True)
                print("[{}/{}] time: {:.4f}s\n\n".format(
                    ii + 1, len(train_dataloader),
                    time.time() - t4epoch))
        # end for epoch
        print(
            "loss: {} | loss_fg: {} | loss_bg:{} | loss_entmin: {} | loss_lap: {}"
            .format(loss.data.item(), loss_fg.data.item(), loss_bg.data.item(),
                    loss_entmin.data.item(), loss_lap.data.item()))
    # end for dataloader
    writer.close()
    print("training was Finished!")
    print("Total time elapsed: {:.0f} h {:.0f} m {:.0f} s\n".format(
        (time.time() - t_start) // 3600, (time.time() - t_start) / 60 % 60,
        (time.time() - t_start) % 60))


def train(**kwargs):
    """
    GCN training
    ---
    - the folder you need:
        - args.path4AffGraph
        - args.path4node_feat
        - path4partial_label
    - these folder would be created:
        - data/GCN_prediction/label
        - data/GCN_prediction/logit
    """
    t_start = time.time()
    # 根据命令行参数更新配置
    args.parse(**kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 把有改動的參數寫到tensorboard名稱上
    comment_init = ''
    for k, v in kwargs.items():
        comment_init += '|{} '.format(v)
    writer = SummaryWriter(comment=comment_init)

    # === set evaluate object for evaluate later
    IoU = IOUMetric(args.num_class)
    IoU_CRF = IOUMetric(args.num_class)

    # === dataset
    train_dataloader = graph_voc()

    # === for each image, do training and testing in the same graph
    # for ii, (adj_t, features_t, labels_t, rgbxy_t, img_name, label_fg_t,
    #          label_bg_t) in enumerate(train_dataloader):
    t4epoch = time.time()
    for ii, data in enumerate(train_dataloader):
        # === use RGBXY as feature
        # if args.use_RGBXY:
        #     data["rgbxy_t"] = normalize_rgbxy(data["rgbxy_t"])
        #     features_t = data["rgbxy_t"].clone()
        # === only RGB as feature
        if True:
            """ is constructing................ """
            H, W, C = data["rgbxy_t"].shape
            A = torch.zeros([H * W, H * W], dtype=torch.float64)

            def find_neibor(card_x, card_y, H, W, radius=2):
                """
                Return idx of neibors of (x,y) in list
                ---
                """
                neibors_idx = []
                for idx_x in np.arange(card_x - radius, card_x + radius + 1):
                    for idx_y in np.arange(card_y - radius,
                                           card_y + radius + 1):
                        if (-radius < idx_x < H) and (-radius < idx_y < W):
                            neibors_idx.append(
                                (idx_x * W + idx_y, idx_x, idx_y))
                return neibors_idx

            t_start = time.time()
            t_start = t4epoch
            neibors = dict()
            for node_idx in range(H * W):
                card_x, card_y = node_idx // W, node_idx % W
                neibors = find_neibor(card_x, card_y, H, W, radius=1)
                # print("H:{} W:{} | {} -> ({},{})".format(
                # H, W, node_idx, card_x, card_y))
                for nei in neibors:
                    # print("nei: ", nei)
                    diff_rgb = data["rgbxy_t"][
                        card_x, card_y, :3] - data["rgbxy_t"][nei[1],
                                                              nei[2], :3]
                    diff_xy = data["rgbxy_t"][card_x, card_y,
                                              3:] - data["rgbxy_t"][nei[1],
                                                                    nei[2], 3:]
                    A[node_idx, nei[0]] = torch.exp(
                        -torch.pow(torch.norm(diff_rgb), 2) /
                        (2. * args.CRF_deeplab["bi_rgb_std"])) + torch.exp(
                            -torch.pow(torch.norm(diff_xy), 2) /
                            (2. * args.CRF_deeplab["bi_xy_std"]))
            print("{:3.1f}s".format(time.time() - t_start))
            D = torch.diag(A.sum(dim=1))
            L_mat = D - A

        # === Model and optimizer
        img_label = load_image_label_from_xml(img_name=data["img_name"],
                                              voc12_root=args.path4VOC_root)
        img_class = [idx + 1 for idx, f in enumerate(img_label) if int(f) == 1]
        num_class = np.max(img_class) + 1
        # debug("num_class: {}  {}".format(num_class + 1, type(num_class + 1)),
        #       line=290)
        model = GCN(
            nfeat=data["features_t"].shape[1],
            nhid=args.num_hid_unit,
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # image label don't have BG
            # adaptive num_class should have better performance
            nclass=args.num_class,  # args.num_class| num_class
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            dropout=args.drop_rate)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        # ==== moving tensor to GPU
        if args.cuda:
            model.to(device)
            data["features_t"] = data["features_t"].to(device)
            data["adj_t"] = data["adj_t"].to(device)
            data["labels_t"] = data["labels_t"].to(device)
            data["label_fg_t"] = data["label_fg_t"].to(device)
            data["label_bg_t"] = data["label_bg_t"].to(device)
            L_mat = L_mat.to(device)

        # === save the prediction before training
        if args.save_mask_before_train:
            model.eval()
            postprocess_image_save(img_name=data["img_name"],
                                   model_output=model(data["features_t"],
                                                      data["adj_t"]).detach(),
                                   epoch=0)

        # ==== Train model
        t4epoch = time.time()
        criterion_ent = HLoss()
        # criterion_sym = symmetricLoss()
        if debug:
            print("fg label: ", torch.unique(data["label_fg_t"]))
            print("bg label: ", torch.unique(data["label_bg_t"]))
            print("img_class: ", img_class)
        for epoch in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()
            output = model(data["features_t"], data["adj_t"])

            # === seperate FB/BG label
            loss_fg = F.nll_loss(output, data["label_fg_t"], ignore_index=255)
            loss_bg = F.nll_loss(output, data["label_bg_t"], ignore_index=255)
            # F.log_softmax(label_fg_t, dim=1)
            loss_entmin = criterion_ent(output,
                                        data["labels_t"],
                                        ignore_index=255)
            # loss_sym = criterion_sym(output, labels_t, ignore_index=255)
            loss_lap = torch.trace(
                torch.mm(output.transpose(1, 0),
                         torch.mm(L_mat.type_as(output), output))) / (H * W)
            gamma = 1e-3
            loss = loss_fg + loss_bg + 10. * loss_entmin + gamma * loss_lap
            # loss = F.nll_loss(output, labels_t, ignore_index=255)

            if loss is None:
                print("skip this image: ", data["img_name"])
                break

            # === for normalize cut
            # lamda = args.lamda
            # n_cut = 0.
            # if args.use_regular_NCut:
            #     W = gaussian_propagator(output)
            #     d = torch.sum(W, dim=1)
            #     for k in range(output.shape[1]):
            #         s = output[idx_test_t, k]
            #         n_cut = n_cut + torch.mm(
            #             torch.mm(torch.unsqueeze(s, 0), W),
            #             torch.unsqueeze(1 - s, 1)) / (torch.dot(d, s))

            # === calculus loss & updated parameters
            # loss_train = loss.cuda() + lamda * n_cut
            loss_train = loss.cuda()
            loss_train.backward()
            optimizer.step()

            # === save predcit mask at max epoch & IoU of img
            if (epoch + 1) % args.max_epoch == 0 and args.save_mask:
                evaluate_IoU(model=model,
                             features=data["features_t"],
                             adj=data["adj_t"],
                             img_name=data["img_name"],
                             epoch=args.max_epoch,
                             img_idx=ii + 1,
                             writer=writer,
                             IoU=IoU,
                             IoU_CRF=IoU_CRF,
                             use_CRF=False,
                             save_prediction_np=True)
                print("[{}/{}] time: {:.4f}s\n\n".format(
                    ii + 1, len(train_dataloader),
                    time.time() - t4epoch))
        # end for epoch
        print(
            "loss: {} | loss_fg: {} | loss_bg:{} | loss_entmin: {} | loss_lap: {}"
            .format(loss.data.item(), loss_fg.data.item(), loss_bg.data.item(),
                    loss_entmin.data.item(), loss_lap.data.item()))
    # end for dataloader
    writer.close()
    print("training was Finished!")
    print("Total time elapsed: {:.0f} h {:.0f} m {:.0f} s\n".format(
        (time.time() - t_start) // 3600, (time.time() - t_start) / 60 % 60,
        (time.time() - t_start) % 60))


if __name__ == "__main__":
    # fire.Fire()
    IOU_dic = {
        .1: 78.07,
        .2: 81.12,
        .3: 81.65,
        .4: 81.38,
        .5: 80.98,
        .7: 79.97,
        .9: 78.83,
        1.: 78.15
    }
    Occupy_dic = {
        .1: 32.1,
        .2: 34.53,
        .3: 36.58,
        .4: 38.32,
        .5: 39.7,
        .7: 41.65,
        .9: 43.30,
        1.: 44.49
    }
    time_now = datetime.datetime.today()
    time_now = "{}_{}_{}_{}h{}m".format(time_now.year, time_now.month,
                                        time_now.day, time_now.hour,
                                        time_now.minute)
    # =======specify the paath and argument ===============
    args.hid_unit = 40
    args.max_epoch = 250
    args.drop_rate = .3
    args.confident_ratio = .3
    # args.path4train_images | args.path4val_images
    args.path4train_images = args.path4train_images
    args.path4AffGraph = os.path.join("irn", "AFF_MAT_normalize_IRNet")

    args.path4partial_label_label = os.path.join(
        args.path4partial_label_label,
        # "RES_CAM_TRAIN_AUG_PARTIAL_PSEUDO_LABEL@PIL_near@confident_ratio_{}_DN"
        "RES_CAM_TRAIN_AUG_PARTIAL_PSEUDO_LABEL@PIL_near@confident_ratio_{}_cam_DN"
        .format(args.confident_ratio))
    args.path4node_feat = os.path.join("irn", "AFF_FEATURE_res50")
    args.use_LP = False
    descript = "dataset: {}, graph: {}, feature: {}, partial label: {}".format(
        os.path.basename(args.path4data), os.path.basename(args.path4AffGraph),
        os.path.basename(args.path4node_feat),
        os.path.basename(args.path4partial_label_label))
    descript = "GCN prediction@IRNet@KNN laplacian @loss_ent@ PPL confident ratio={} & IOU@{} occupy@{}".format(
        args.confident_ratio, IOU_dic[args.confident_ratio],
        Occupy_dic[args.confident_ratio])
    print("descript ", descript)
    print("here is branch `debug` !!")
    # args.path4prediction_np = os.path.join(args.path4prediction_np,
    #                                        args.path4partial_label_label)
    args.path4GCN_label = os.path.join(args.path4GCN_label, time_now)
    args.path4GCN_logit = os.path.join(args.path4GCN_logit, time_now)
    # ====  training  =======
    train(use_crf=False, descript=descript)
    evaluate_dataset_IoU(file_list=args.path4train_images,
                         predicted_folder=args.path4GCN_label,
                         descript=descript)