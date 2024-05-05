# Convergence analysis of deep residual networks
This is the code implementation of Convergence analysis of deep residual networks (https://www.worldscientific.com/doi/10.1142/S021953052350029X).

## Prerequisites
Our implementation is tested under following settings:
    
- CUDA version: V11.2
- Python version: 3.10.4
- Python libs: keras==2.8.0, tensorflow==2.83, torch==2.0.1, torchvision==0.15.2, numpy==1.25.1, matplotlib==3.7.2

## Usage
Notice that the parameter $\alpha$ in our paper is named as `gamma` in our code implementations.

### Mathematical verification related
If one wants to reproduce the experiments in section 7.2 of our paper, one could run `python test_theorem.py`.

The default settings of the random resnet is as the same as the settings in our paper, one could adjust the settings in `./test_theorem.py` (row 32 -- 44).

The results (figure of differences) will be automatically stored in `./results/test_theorem/test_plot.pdf`.

### CIFAR-10 related
For training model for CIFAR-10, one could set the configurations of resnet in `./test_new_ResNet.py` and run `python test_new_ResNet.py`.

The trained model and related information will be automatically stored in directory `./results/result_d_x_m_y_timestamp/`, where x, y and timestamp denote depth, width and time of the experiment respectively.

For checking the results and plotting the sum of the weights, one could set the path of stored results in `./test_ResNet_model.py` (row 21) and run `python test_ResNet_model.py`. And the figures will be automatically stored in the same directories of results.

The model we trained are stored in `./result/result_d_75_m_128_2023_09_15_08_39_24/`.
