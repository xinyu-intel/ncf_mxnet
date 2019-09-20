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

We provide the processed datasets on [Google Drive](https://drive.google.com/drive/folders/1qACR_Zhc2O2W0RrazzcepM2vJeh0MMdO?usp=sharing): MovieLens 20 Million (ml-20m), you can download directly or 
run the script to prepare the datasets:
```
python convert.py 
```

train.rating: 
- Train file (positive instances).
- Each Line is a training instance: userID\t itemID\t 

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t 

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 999 negative samples.  
- Each line is in the format: userID,\t negativeItemID1\t negativeItemID2 ...


## Training

Currently doesn't support train neumf model with pre-trained gmf and mlp model.

### ml-20m

```
# train neumf on ml-20m dataset
python train_ncf.py --batch-size 65536 --learning-rate 0.0002 
# train gmf on ml-20m dataset
python train_ncf.py --batch-size 65536 --learning-rate 0.0002  --model-type='gmf'
# train mlp on ml-20m dataset
python train_ncf.py --batch-size 65536 --learning-rate 0.0002  --model-type='mlp'
```

## Inference

```
# neumf inference on ml-20m dataset
python eval_ncf.py  --epoch=2 --deploy --prefix=./model/ml-20m/neumf
```

# Calibration

```
# neumf calibration on ml-20m dataset
python eval_ncf.py --epoch=2 --deploy --prefix=./model/ml-20m/neumf --calibration
# neumf int8 inference on ml-20m dataset
python eval_ncf.py --epoch=2 --deploy --prefix=./model/ml-20m/neumf-quantized
```

## Evaluate Accuracy

```
# evaluate neumf accuracy on ml-20m dataset
python eval_ncf.py --epoch=2 --evaluate 
```
