## Folder structure
- /home/User
    - pygcn
        - data
            - Complete_pseudo_label
                - label
                - logit
            - deeplab_prediction
                - label
                - logit
            - GCN_prediction
                - label
                - logit
            - partial_pseudo_label
                - label
                - logit
        - runs
        - submitVOC
    - psa
        - AFF_MAT_normalize
        - AFF_FEATURE_res38
        - RES38_PARTIAL_PSEUDO_LABEL_DN
    - semi_VCIP2020
        - data
            - datasets
            - models\coco\deeplabv1_resnet101\caffemodel\deeplabv1_resnet101-coco.pth
        - deeplab
        - libs
            - models
            - utils
        - deeplab_res101_ImageNetPretrain

## Enviroment
- pytorch 1.4.0
- scipy 1.2.0
- opencv
- pydensecrf.densecrf 
- PIL
- fire
# [PSA](https://github.com/jiwoon-ahn/psa)
- [the weight of classifier](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view?usp=sharing)
- [the weight of affinityNet](https://drive.google.com/open?id=1mFvTH3siw0SS0vqPH0o9N3cI_ISQacwt)

 `cd psa`  # change your path
## generate CAM
`python infer_cls.py`

## generate Graph and node feature (you have to generate CAM first)
`python infer_aff.py`

# GCN

- `cd pygcn`  # change your path
-  to train GCN you need:
    1. partial pseudo label
    2. affinity matrix(Graph)
    3. node feature

## generate partial pseudo label
- `python make_dataset.py` # generate 3 folders of data for training of GCN

## GCN training + evaluation 
- `python train2020.py`

## apply denseCRF to GCN prediction
Apply dense CRF in pred_root (default: the least modify folder in `data/GCN_prediction/logit`)
>`python CRF.py apply --CRF_parameter=args.CRF`   

- parameter:
    - --`CRF_parameter`: args.CRF | args.CRF_deeplab | args.CRF_psa  # default: args.CRF
    - --`path4saveCRF_label`: default: `data/Complete_pseudo_label/label`
    - --`path4saveCRF_logit`: default: `data/Complete_pseudo_label/logit/yyyy_mm_dd_hhmm`
    - --`f_list`: the `.txt` for VOC2012 dataset (e.g. train.txt)
    - --`pred_root`: the folder you want to apply CRF
    - --`mode`: other | rw | deeplab |  this funtion is still construct...

## Apply dense CRF on Deeplab output or random walk
Apply dense CRF in pred_root (default: the least modify folder in `semi_VCIP2020/data/logit`)
>`python utils.py apply_dCRF --mode=deeplab --save_path=data/deeplab_prediction/DeepLab@Psa@CRF_DeepLabSeeting@CompletePseudoLabel_IOU57.07
`
- parameter:
    - --`mode`: deeplab | rw  (default: deeplab)
    - --`src`: the path you want to apply dense CRF (default: the least modify folder in `semi_VCIP2020/data/logit`)
    - --`save_path`: the path you want to save the result of CRF 
    - --`dataset`: test | val | train | train_aug (default: val)
    - --`user`:  (default: Ours)

## Evaluate GCN
To evaluate the GCN

Example
>`python utils.py evaluate_dataset_IoU --predicted_folder="data/GCN_prediction/label" --file_list="../psa/voc12/train.txt" --descript="PPL iou 81.99@train"`
- parameter:
    - --`predicted_folder`: the path you want to evaluate
    - --`file_list`: the dataset you want to use to evaluate (default: `"../psa/voc12/train.txt"`)
    - --`descript`: the path you want to save the result of CRF 

# DeepLab v2
 `cd semi_VCIP2020`  # change your path
## Train
>`python main.py train`

## Testing
>`python main.py test --[arg]=[arg Value]`
- arg:
    - model_iteration
    - save_class

## Parameter
- `confings/voc12.yaml`  # directly change the value in `voc12.yaml`
