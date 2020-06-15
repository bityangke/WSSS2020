## Folder structure
- /home/User
    - pygcn
        - ~~data_v8~~
        - predict_result_matrix_visual_new
            - 250
        - predict_result_visual_epochs
            - 250
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
# PSA
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
- `python CRF.py apply`
- `python CRF.py help` # show more detail
