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
python `infer_cls.py`

## generate Graph and node feature (you have to generate CAM first)
python `infer_aff.py`

# GCN

- `cd pygcn`  # change your path
-  to train GCN you need:
    1. partial pseudo label
    2. affinity matrix(Graph)
    3. node feature
## GCN training + evaluation 
- python `train2020.py`

## generate partial pseudo label
- python `make_dataset.py`

