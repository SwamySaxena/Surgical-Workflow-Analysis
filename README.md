# Surgical Workflow Analysis Using Temporal Convolution Networks

## Introduction

The is an experimentation based on this [paper: Free Lunch for Surgical Video Understanding by Distilling Self-Supervisions]([http://camma.u-strasbg.fr/datasets](https://arxiv.org/abs/2205.09292))

We create a pipeline consisting of a resnet50 model acting as a feature extractor followed by a TCN model acting as the predictor.

Framework visualization


## Preparation

## Data Preparation
* We use the dataset [Cholec80](http://camma.u-strasbg.fr/datasets) and extract 25 frames per second per video. Further, the frames are resized to 250*250.
* Training and test data split
   Cholec80: first 40 videos for training and the rest 40 videos for testing.

##
* Video wise features are extracted from framewise features that resnet50 extracts.
* These features are then used to train the modified TCN model.

* The base TCN model is as follows : 
![img1](https://github.com/SwamySaxena/Surgical-Workflow-Analysis/blob/main/img1.jpg)

*The modified structure looks as follows:
![img2](https://github.com/SwamySaxena/Surgical-Workflow-Analysis/blob/main/img2.jpg)

And it utilizes these:
![img3](https://github.com/SwamySaxena/Surgical-Workflow-Analysis/blob/main/img3.jpg)





