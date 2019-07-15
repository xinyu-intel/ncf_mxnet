# Neural Collaborative Filtering

This is MXNet implementation for the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling. 

Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

Code Reference: https://github.com/hexiangnan/neural_collaborative_filtering

## Environment Settings
We use MXnet with MKL-DNN as the backend. 
- MXNet version:  '1.5.0rc2'

## Install
```
pip install -r requirements.txt
```

## Dataset

We provide two processed datasets on [Google Drive](https://drive.google.com/drive/folders/1qACR_Zhc2O2W0RrazzcepM2vJeh0MMdO?usp=sharing): MovieLens 1 Million (ml-1m) and MovieLens 20 Million (ml-20m).

train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

It will take long time to prepare `test.negative` dataset, you can run the following command to make one if you want.

```
python convert.py --dataset='ml-20m' --negative-num=99
```

## Training

Currently doesn't support train neumf model with pre-trained gmf and mlp model.

### ml-1m

```
# train neumf on ml-1m dataset
python train_ncf.py --dataset='ml-1m' --layers='[64,32,16,8]' --factor-size-gmf=8
# train gmf on ml-1m dataset
python train_ncf.py --dataset='ml-1m' --factor-size-gmf=8 --model-type='gmf'
# train mlp on ml-1m dataset
python train_ncf.py --dataset='ml-1m' --layers='[64,32,16,8]' --model-type='mlp'
```

### ml-20m

```
# train neumf on ml-20m dataset
python train_ncf.py --dataset='ml-20m' --layers='[256, 128, 64]' --factor-size-gmf=64
# train gmf on ml-20m dataset
python train_ncf.py --dataset='ml-20m' --factor-size-gmf=64 --model-type='gmf'
# train mlp on ml-20m dataset
python train_ncf.py --dataset='ml-20m' --layers='[256, 128, 64]' --model-type='mlp'
```

## Inference

```
# neumf inference on ml-20m dataset
python eval_ncf.py --batch-size=256 --dataset='ml-20m' --layers='[256, 128, 64]' --factor-size-gmf=64 --epoch=0
```

## Evaluate Accuracy

```
# evaluate neumf accuracy on ml-20m dataset
python eval_ncf.py --batch-size=256 --dataset='ml-20m' --layers='[256, 128, 64]' --factor-size-gmf=64 --epoch=0 --evaluate 
```