import warnings
import torch as t
import os
import getpass


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097  # visdom 端口

    # train_data_root = os.path.join('data', 'train')  # 训练集存放路径
    # test_data_root = os.path.join('data', 'test1')  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 50  # print info every N batch

    debug_file = os.path.join(
        'tmp', 'debug')  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
    alpha4LP = .4
    save_mask_before_train = False
    use_gaussian_propagator = False
    use_Biliteral_graph = False
    use_RGBXY = False
    use_LP = False
    # ====parameter for GCN =============================================
    lr = .01
    weight_decay = 5e-4
    seed = 42  # Random seed
    sigma = 1.  # type=float,gaussian filter parameer
    lamda = 0.  # type=float, loss function parameter ( of background points)
    num_class = 21
    max_epoch = 250  # 200
    num_hid_unit = 40  # 80
    drop_rate = .3  # .5
    cuda = t.cuda.is_available()
    use_regular_NCut = False

    # == parameter for pre process =====================================
    store_graph = False  # type=bool
    thre4rw = .8  # type=float
    thre4confident_region = .6  # type=float
    path4PseudoLabel_LA = os.path.join("..", "psa", "VGG_LA_CRF")
    path4PseudoLabel_HA = os.path.join("..", "psa", "VGG_HA_CRF")
    path4Image = os.path.join("..", "psa", "VOC2012", "JPEGImages")
    path4VOC = path4Image
    path4Class = os.path.join("..", "psa", "VOC2012", "SegmentationClassAug")
    path4VOC_class_aug = os.path.join("..", "psa", "VOC2012",
                                      "SegmentationClassAug")
    path4VOC_root = os.path.join("..", "psa", "VOC2012")
    path4PsaFeature = os.path.join("..", "psa", "aff_map")
    pseudo_label_thre = .95  # threshold for select confident label
    path4data = "data"
    path4CAM = os.path.join("..", "psa", "RES_CAM__")
    path4Pseudo_label = os.path.join("..", "psa", "RES_PSEUDO_LABEL(2020)")
    # ===paeameter for dataset =====================================
    graph_path = dict()
    graph_path['AFF'] = os.path.join("..", "psa", "aff_matrix")
    graph_path['GT'] = os.path.join("..", "psa", "graph-full")
    graph_path['RW'] = os.path.join("..", "psa", "VGG_RAM__")

    graph_pre = dict()
    graph_pre['AFF'] = graph_pre['RW'] = ''
    graph_pre['GT'] = 'ind.'

    graph_ext = dict()
    graph_ext['AFF'] = ".npy"
    graph_ext['GT'] = ".graph"
    graph_ext['RW'] = ".pkl"

    start_data_idx = 0
    end_data_idx = -1
    # aff_matrix is graph, aff_map is node feature
    path4AffGraph = os.path.join("..", "..", "..", "work", getpass.getuser(),
                                 "psa", "aff_matrix")
    # path4node_feat = os.path.join("..", "psa", "AFF_FEATURE_res38")
    path4node_feat = "../psa/AFF_FEATURE_res38"
    path4train_images = os.path.join("..", "psa", "voc12", "train.txt")
    path4train_aug_images = os.path.join("..", "psa", "voc12", "train_aug.txt")
    path4val_images = os.path.join("..", "psa", "voc12", "val.txt")
    path4trainval_images = os.path.join("..", "psa", "voc12", "trainval.txt")
    path4partial_label = "../psa/RES38_PARTIAL_PSEUDO_LABEL_DN"
    # ===paeameter for post process =====================================
    # path4save_img = os.path.join('predict_result_visual_epochs')
    path4save_img = os.path.join("..", "predict_result_visual_epochs")
    path4save_LP = os.path.join("..", "LP_prediction")
    save_prediction_np = True
    path4prediction_np = "predict_result_matrix_visual_new"
    # path4prediction_np = os.path.join("..", "predict_result_matrix_visual_new")
    save_mask = True
    path4img = os.path.join("..", "psa", "VOC2012", "JPEGImages")
    output_rate = 8
    path4saveInfo = 'evaluation4dataset.md'
    use_label_propagation = False
    use_crf = False
    crf_scale_factor = 1
    crf_num_iteration = 10
    path4saveCRF = 'predict_result_CRF'
    path4saveCRF_np = 'predict_result_CRF_npy'
    # === CRF
    CRF_deeplab = dict()
    CRF_deeplab["iter_max"] = 10
    CRF_deeplab["pos_w"] = 3
    CRF_deeplab["pos_xy_std"] = 1
    CRF_deeplab["bi_w"] = 4
    CRF_deeplab["bi_xy_std"] = 67
    CRF_deeplab["bi_rgb_std"] = 3
    CRF_psa = dict()
    CRF_psa["iter_max"] = 10
    CRF_psa["pos_w"] = 3
    CRF_psa["pos_xy_std"] = 3
    CRF_psa["bi_w"] = 10
    CRF_psa["bi_xy_std"] = 80
    CRF_psa["bi_rgb_std"] = 13
    CRF = dict()
    CRF["iter_max"] = 10
    CRF["pos_w"] = 3
    CRF["pos_xy_std"] = 4
    CRF["bi_w"] = 3
    CRF["bi_xy_std"] = 83
    CRF["bi_rgb_std"] = 5

    # ===================================================================

    def parse(self, **kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut:  {}".format(k))
            else:
                setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # print('user config:')
        # for k, v in self.__class__.__dict__.items():
        #     if not k.startswith('_'):
        #         print(k, getattr(self, k))


opt = DefaultConfig()
