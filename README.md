# TimeGAN
A pytorch implementation of  Time-series Generative Adversarial Networks (https://github.com/jsyoon0823/TimeGAN) 

## Project Description
The Goal was to create smoothed time series data via a GAN. This should be achieved via a combination of https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf and https://arxiv.org/pdf/2106.10414.pdf in PyTorch.

## Summary 
In this updated Version I was able to achieve a better recreation of the original time GAN implementation. With increasing size of epochs, the quality (evauluated on the third method of the time gan paper, namely pca and tsne plots) seems to increase. With epochs = 20 I was able to achieve these results: 


Also an Ada FNN Layer was added to the Generator to try if the generated time Series will smooth out. This seems to be the case, however during some test runs without the ada fnn lyer included, smooth(er) curves were also generated, so it is hard to say how beneficial the ada fnn layer is in fact. See the demo notebook for visualizations. 

## Repository Structure: 
requirements.txt contains all dependencies and can be run with pip: 
pip install -r requirements.txt 
utils.py contains all helper functions, mostly from the original repository
modules_and_training contains the main implementation of TimeGAN. The Network blocks are defined there, aswell as a function that runs training and returns the trained networks. 
Demo_Time_GAN_and_ADA_fnn includes a demo run of the time gan without ada fnn layer as well as a demo run with a ada fnn layer.  
Tests.py contains test that were written to ensure correct functioning of each module.
