# Differential Privacy Protection against Membership Inference Attack on Genomic Data

## Introduction
Machine learning is powerful to model massive genomic data while genome privacy is a growing concern. Studies have shown that not only the raw data but also the trained model can potentially infringe genome privacy. An example is the membership inference attack (MIA), by which the adversary, who only queries a given target model without knowing its internal parameters, can determine whether a specific record was included in the training dataset of the target model. 
Differential privacy (DP) has been used to defend against MIA with rigorous privacy guarantee. In this paper, we investigate the vulnerability of machine learning against MIA on genomic data, and evaluate the effectiveness of using DP as a defense mechanism. We consider two widely-used machine learning models, namely Lasso and convolutional neural network (CNN), as the target model. We study the trade-off between the defense power against MIA and the prediction accuracy of the target model under various privacy settings of DP. Our results show that the relationship between the privacy budget and target model accuracy can be modeled as a log-like curve, thus a smaller privacy budget provides stronger privacy guarantee with the cost of losing more model accuracy. We also investigate the effect of model sparsity on model vulnerability against MIA. Our results demonstrate that in addition to prevent overfitting, model sparsity can work together with DP to significantly mitigate the risk of MIA.

## Differential Privacy (DP)
DP is a state-of-the-art privacy protection standard. It requires that a mechanism outputting information about an underlying dataset is robust to any change of one sample.  It has been shown that DP can be an effective solution for granting wider access to machine learning models and results while keep them private. To investigate the impact of DP on the utility of machine learning models, we incorporate DP to two machine learning methods, namely Lasso and Convolutional neural network (CNN), that are commonly used for genomic data analysis. 

The corresponding code is in folder of differential-privacy.
```
Differential Privacy
|-- Lasso-dp.py
|-- CNN-dp.py
```

## Membership Inference Attack
Membership inference attack (MIA) is a privacy-leakage attack that predicts whether a given record was used in training a target model. It works under a setting where the target model is opaque but remotely accessible. It predicts whether a given record was in the model’s training dataset based on the output of the target model for the given record.

The corresponding code is in folder of membership-inference-attack.
```
Membership Inference Attack
|-- Lasso-MIA.py
|-- CNN-MIA.py
```

## Getting Started

### Prerequisites
```
Python >= 3.6 
virtualenv >= 16.4.3
```
### Setup
1. Create virtual environment
```
git clone https://github.com/shilab/DP-MIA.git
cd DP-MIA/
mkdir venv
python3 -m venv venv/
source venv/bin/activate
```
2. Install requirement dependents
```
pip install tensorflow==1.14 tensorflow_privacy sklearn pandas jupyter mia
```

## Citation
Junjie Chen, Hui Wang, Xinghua Shi, Differential Privacy Protection against Membership Inference Attack on Machine Learning for Genomic Data, in submission.
