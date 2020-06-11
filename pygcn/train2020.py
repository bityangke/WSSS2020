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
from utils import IOUMetric, colors_map, evaluate_dataset_IoU, saveInfo
from utils import load_image_label_from_xml, CLS_NAME_TO_ID, CLS_ID_TO_NAME
from utils import ANNOT_FOLDER_NAME, SEG_NAME_TO_ID, SEG_ID_TO_NAME
from utils import crf_inference_inf, load_img_name_list, show_timing
from utils import HLoss, symmetricLoss

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
    # === in os.path.join(args.path4save_img, str(epoch) )
    if save_prediction_np:
        path = os.path.join(args.path4prediction_np, str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
            print("make folder:", path)
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

    if save_CRF:
        crf_score = crf_inference_inf(img_name=img_name,
                                      img=img,
                                      probs=up_predict.cpu().numpy(),
                                      labels=up_predict.shape[0])
        up_crf_mask = np.argmax(crf_score, axis=0)
        scipy.misc.toimage(up_crf_mask,
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(args.path4save_img, "CRF",
                                            img_name + '.png'))

    # === save the prediction as label
    # === in os.path.join(path4save, img_name + '.png')
    path4save = os.path.join(args.path4save_img, str(epoch))
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


def gaussian_propagator_sparse(features):
    """
    Calculus gaussian kernal propagator for label propagation 
    ---
    features: a [num_nodes, len_feature] tensor
    e.g. for 10 nodes RGBXY the features.shape=(10,5)
    return the tensor (cpu)
    # """
    pass
    # output = features.cpu()  # [n,f]
    # num_nodes, len_feature = output.shape
    # rgbxy_repeate = output.repeat(num_nodes, 1)  # [n,f]
    # rgbxy_np = output.numpy()
    # rgbxy = torch.from_numpy(rgbxy_np.repeat(num_nodes, 0))  # .cuda()
    # rgbxy = rgbxy - rgbxy_repeate
    # rgbxy = rgbxy.permute(1, 0)
    # rgbxy = rgbxy.view(len_feature, num_nodes,
    #                    num_nodes)  # 5 is the dim of feature (i.e. rgbxy)
    # # variance was already absorted to RGBXY, so we don't devide variance here
    # S = torch.exp(-torch.pow(torch.norm(rgbxy, dim=0), 2) / 2.)
    # return S.cuda()


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
        os.path.join(args.path4save_img, str(epoch), img_name + '.png'))
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

    # === evaluate IoU_CRF
    if use_CRF:
        mask_crf = Image.open(
            os.path.join(args.path4save_img, 'CRF', img_name + '.png'))
        mask_crf = np.asarray(mask_crf)
        # model prediction with CRF
        mask_predit_CRF = Image.open(
            os.path.join(args.path4save_img, 'label_propagation',
                         img_name + '.png'))
        mask_predit_CRF = np.asarray(mask_predit_CRF)
        IoU_CRF.add_batch(mask_crf, mask_gt)
        acc_CRF, acc_cls, iu, mean_iu_tensor_CRF, fwavacc = IoU_CRF.evaluate()
        # === write in tensorboard ===
        writer.add_scalar("IoU_with_CRF",
                          mean_iu_tensor_CRF.cpu().numpy(),
                          global_step=img_idx)
        print("Acc_CRF: {:>7.2f} IoU_CRF: {:>7.2f}%\n".format(
            acc_CRF,
            mean_iu_tensor_CRF.cpu().numpy() * 100))


def debug(message="", line=None):
    if line:
        print("[{}] ".format(line), end=' ')
    print(message)
    input("debug---->")


def train(**kwargs):
    t_start = time.time()
    # 根据命令行参数更新配置
    args.parse(**kwargs)

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
    for ii, (adj_t, features_t, labels_t, rgbxy_t, img_name, label_fg_t,
             label_bg_t, idx_train_fg_t,
             idx_train_bg_t) in enumerate(train_dataloader):
        # === use RGBXY as feature
        if args.use_RGBXY:
            rgbxy_t = normalize_rgbxy(rgbxy_t)
            features_t = rgbxy_t.clone()

        # === build graph by rgbxy feature
        # === transform shape [H,W,5] to [5,H,W]

        # === only RGB as feature
        # adj_t = gaussian_propagator(
        #     features=rgbxy_t.permute(2, 0, 1)[:3, :, :])

        # === Model and optimizer
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        img_class = [idx + 1 for idx, f in enumerate(img_label) if int(f) == 1]
        num_class = np.max(img_class) + 1
        # debug("num_class: {}  {}".format(num_class + 1, type(num_class + 1)),
        #       line=290)
        model = GCN(
            nfeat=features_t.shape[1],
            nhid=args.num_hid_unit,
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # image label don't have BG
            # adaptive num_class should have better performance
            nclass=num_class,  # args.num_class| num_class
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            dropout=args.drop_rate)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        # ==== moving tensor to GPU
        if args.cuda:
            model.cuda()
            features_t = features_t.cuda()
            adj_t = adj_t.cuda()
            labels_t = labels_t.cuda()
            idx_train_fg_t = idx_train_fg_t.cuda()
            idx_train_bg_t = idx_train_bg_t.cuda()
            label_fg_t = label_fg_t.cuda()
            label_bg_t = label_bg_t.cuda()
            # idx_test_t = idx_test_t.cuda()  # bool

        # === save the prediction before training
        if args.save_mask_before_train:
            model.eval()
            postprocess_image_save(img_name=img_name,
                                   model_output=model(features_t,
                                                      adj_t).detach(),
                                   epoch=0)

        # ==== Train model
        t4epoch = time.time()
        criterion_ent = HLoss()
        criterion_sym = symmetricLoss()
        print("fg label: ", torch.unique(label_fg_t))
        print("bg label: ", torch.unique(label_bg_t))
        print("img_class: ", img_class)
        for epoch in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()
            output = model(features_t, adj_t)
            # print("label_fg_t.min()",label_fg_t.min())
            # print("output.argmax(dim=1)",output.argmax(dim=1))
            # input("[336]--------<")
            # === seperate FB/BG label
            loss_fg = F.nll_loss(output, label_fg_t, ignore_index=255)
            loss_bg = F.nll_loss(output, label_bg_t, ignore_index=255)
            # F.log_softmax(label_fg_t, dim=1)
            # loss_entmin = criterion_ent(output, labels_t, ignore_index=255)
            # loss_sym = criterion_sym(output, labels_t, ignore_index=255)
            loss = loss_fg + loss_bg

            # loss = F.nll_loss(output, labels_t, ignore_index=255)

            if loss is None:
                print("skip this image: ", img_name)
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
            # print(
            #     "loss: {} | loss_fg: {} | loss_bg:{} | loss_entmin: {}".format(
            #         loss.data.item(), loss_fg.data.item(), loss_bg.data.item(),
            #         loss_entmin.data.item()))
            # === save predcit mask of LP at max epoch & IoU of img
            if (epoch + 1) % args.max_epoch == 0 and args.save_mask:
                evaluate_IoU(model=model,
                             features=features_t,
                             adj=adj_t,
                             img_name=img_name,
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
    # end for dataloader
    writer.close()
    print("training was Finished!")
    print("Total time elapsed: {:.0f} h {:.0f} m {:.0f} s\n".format(
        (time.time() - t_start) // 3600, (time.time() - t_start) / 60 % 60,
        (time.time() - t_start) % 60))
    saveInfo(predicted_folder=os.path.join(args.path4save_img,
                                           str(args.max_epoch)),
             method='GCN_LP')


def train_laplacian(**kwargs):
    """
    Additional regularization loss was add to GCN (6.10)
    ---
    - RGB feature
    - KNN Graph with RBF ,k=8
    - Region Growing
    """
    t_start = time.time()
    # 根据命令行参数更新配置
    args.parse(**kwargs)

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
    for ii, (adj_t, features_t, labels_t, rgbxy_t, img_name, label_fg_t,
             label_bg_t, idx_train_fg_t,
             idx_train_bg_t) in enumerate(train_dataloader):

        # === build graph by rgbxy feature
        if args.use_Biliteral_graph:
            adj_t = gaussian_propagator(features=rgbxy_t)

        # === Model and optimizer
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        num_class = np.max(
            [idx for idx, f in enumerate(img_label) if int(f) == 1])
        # debug("num_class: {}  {}".format(num_class + 1, type(num_class + 1)),
        #       line=290)
        model = GCN(
            nfeat=features_t.shape[1],
            nhid=args.num_hid_unit,
            nclass=args.num_class,  # image label don't have BG
            dropout=args.drop_rate)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        # ==== moving tensor to GPU
        if args.cuda:
            model.cuda()
            features_t = features_t.cuda()
            adj_t = adj_t.cuda()
            labels_t = labels_t.cuda()
            idx_train_fg_t = idx_train_fg_t.cuda()
            idx_train_bg_t = idx_train_bg_t.cuda()
            label_fg_t = label_fg_t.cuda()
            label_bg_t = label_bg_t.cuda()
            # idx_test_t = idx_test_t.cuda()  # bool

        # === save the prediction before training
        if args.save_mask_before_train:
            model.eval()
            postprocess_image_save(img_name=img_name,
                                   model_output=model(features_t,
                                                      adj_t).detach(),
                                   epoch=0)

        # ==== Train model
        t4epoch = time.time()
        criterion_ent = HLoss()
        criterion_sym = symmetricLoss()
        print("fg label: ", torch.unique(label_fg_t))
        for epoch in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()
            output = model(features_t, adj_t)
            # print("label_fg_t.min()",label_fg_t.min())
            # print("output.argmax(dim=1)",output.argmax(dim=1))
            # input("[336]--------<")
            # === seperate FB/BG label
            loss_fg = F.nll_loss(output, label_fg_t, ignore_index=255)
            loss_bg = F.nll_loss(output, label_bg_t, ignore_index=255)
            # F.log_softmax(label_fg_t, dim=1)
            # loss_entmin = criterion_ent(output, labels_t, ignore_index=255)
            # loss_sym = criterion_sym(output, labels_t, ignore_index=255)
            loss = loss_fg + loss_bg

            # loss = F.nll_loss(output, labels_t, ignore_index=255)

            if loss is None:
                print("skip this image: ", img_name)
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
            # print(
            #     "loss: {} | loss_fg: {} | loss_bg:{} | loss_entmin: {}".format(
            #         loss.data.item(), loss_fg.data.item(), loss_bg.data.item(),
            #         loss_entmin.data.item()))
            # === save predcit mask of LP at max epoch & IoU of img
            if (epoch + 1) % args.max_epoch == 0 and args.save_mask:
                evaluate_IoU(model=model,
                             features=features_t,
                             rgbxy=rgbxy_t,
                             adj=adj_t,
                             img_name=img_name,
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
    # end for dataloader
    writer.close()
    print("training was Finished!")
    print("Total time elapsed: {:.0f} h {:.0f} m {:.0f} s\n".format(
        (time.time() - t_start) // 3600, (time.time() - t_start) / 60 % 60,
        (time.time() - t_start) % 60))
    saveInfo(predicted_folder=os.path.join(args.path4save_img,
                                           str(args.max_epoch)),
             method='GCN_LP')


if __name__ == "__main__":
    # fire.Fire()

    # =======specify the paath and argument ===============
    path4AFF = "AFF_MAT_normalize"  # "aff_map_normalize" | "AFF_MAT_normalize"
    args.parse(
        path4data="data_v8",
        # data_RES_UP_CRF_DN_TRAIN| data_RES_UP_CRF_DN | data_RES_UP_CRF_DN_VAL
        hid_unit=40,
        max_epoch=250,
        drop_rate=.3,
        # path4AffGraph=os.path.join("..", "..", "..", "work", getpass.getuser(),
        #                            "psa", path4AFF),
        path4AffGraph=os.path.join("..", "..", "..", "work", getpass.getuser(),
                                   "psa", path4AFF),
        path4train_images=args.path4train_images,
        path4partial_label=os.path.join(
            "..", "psa", "RES_CAM_TRAIN_AUG_PARTIAL_PSEUDO_LABEL_DN_ASUS"),
        path4node_feat=os.path.join("..", "psa", "AFF_FEATURE_res38"),
        use_LP=False)  # args.path4train_images | args.path4val_images
    descript = "+LP dataset: {}, graph: {}, feature: {}, partial label: {}".format(
        os.path.basename(args.path4data), os.path.basename(args.path4AffGraph),
        os.path.basename(args.path4node_feat),
        os.path.basename(args.path4partial_label))
    print("descript ", descript)
    print("here is branch `debug` !!")
    # ====  training  =======
    train(use_crf=False, descript=descript)
    evaluate_dataset_IoU(predicted_folder=os.path.join(args.path4save_img,
                                                       "250"),
                         descript=descript)
