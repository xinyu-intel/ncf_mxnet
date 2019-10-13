# Neural Collaborative Filtering

This is MXNet implementation for the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling. 

Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

Code Reference: https://github.com/hexiangnan/neural_collaborative_filtering

## Environment Settings
We use MXnet with MKL-DNN as the backend. 
- MXNet version:  '1.5.1'

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

train-ratings.csv
- Train file (positive instances).
- Each Line is a training instance: userID\t itemID\t 

test-ratings.csv
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t 

test-negative.csv
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 999 negative samples.  
- Each line is in the format: userID,\t negativeItemID1\t negativeItemID2 ...

## Pre-trained models

We convert pre-trained NCF model from [MLPerf Pytorch version](https://github.com/mlperf/training/blob/948db9b11cdfa7d953769e53c560396f41617f1b/recommendation/pytorch/).

Pre-trained Pytorch model can be get from [Google Drive](https://drive.google.com/drive/folders/1qACR_Zhc2O2W0RrazzcepM2vJeh0MMdO?usp=sharing) and you can convert it to MXNet by the following command.

```
python export/export.py
```

|dtype|HR|NDCG|
|:---:|:--:|:--:|
|float32|0.6327|0.3809|
|int8|0.6330|0.3812|

## Training

TBD

# Calibration

```
# neumf calibration on ml-20m dataset
python ncf.py --deploy --prefix=./model/ml-20m/neumf --calibration
```

## Inference

```
# neumf float32 inference on ml-20m dataset
python ncf.py --deploy --num-valid=138493 --batch-size=1000 --prefix=./model/ml-20m/neumf
# neumf int8 inference on ml-20m dataset
python ncf.py --deploy --num-valid=138493 --batch-size=1000 --prefix=./model/ml-20m/neumf-quantized
```

## Benchmark

```
# neumf float32 benchmark on ml-20m dataset
python ncf.py --deploy --num-valid=138493 --batch-size=1000 --prefix=./model/ml-20m/neumf --benchmark
# neumf int8 benchmark on ml-20m dataset
python ncf.py --deploy --num-valid=138493 --batch-size=1000 --prefix=./model/ml-20m/neumf-quantized --benchmark
```
