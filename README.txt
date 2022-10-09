This code is used for training content-based story evaluation model in our paper.

## run the experiment
python train.py

but noticing that we have truncated the training data due to the upload size limit.

## error
Our code can successfully run on NVIDIA A100,
but report an unexpected error on NVIDIA V100.
We are still trying to figure out the reason.


## package requirement:
torch                         1.9.0+cu111
torch-cluster                 1.5.9
torch-geometric               1.7.2
torch-geometric-temporal      0.37
torch-scatter                 2.0.7
torch-sparse                  0.6.10
torch-spline-conv             1.2.1
torchaudio                    0.9.0
torchmetrics                  0.2.0
torchtext                     0.9.1
torchvision                   0.10.0+cu111
tqdm                          4.41.1
scikit-learn                  0.20.0
scipy                         1.4.1
python                        3.8