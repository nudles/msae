This file includes instructions for running the multi-modal stacked auto-encoders, denoted as MSAE.
MSAE is implemented in Python (2.7). Basic Python libraries such as numpy and scipy, are required. 
It has been tested on CentOS 6.4 using CUDA 5.0 with NVIDIA GPU (GeForce GTX TITAN).


1. install dependent libraries.
    1.1 install CUDA tookit (https://developer.nvidia.com/cuda-toolkit-50-archive)
    1.2 install CUDAMat (https://code.google.com/p/cudamat/)
        add libcudamat.so to your LD_LIBARAY_PATH
    1.3 install gnumpy (http://www.cs.toronto.edu/~tijmen/gnumpy.html)
        it is already included. Do not use the one from the website directly, 
        as it does not support function pow(matrix, matrix).
    1.4 install configparser 3.3.0r2 (https://pypi.python.org/pypi/configparser)
        this package is an upgrade of configparser built-in with python 2.7

2. prepare training, validation and test data.
    for small training dataset, e.g., Wiki and NUS-WIDE, normalize the image
        features by ZCA. For long text input, normalize it by log(1+x)
    for large training dataset, e.g., Flickr1M, compute the mean and std of
        each feature dimension. normalization will be conducted online during training

    content of each input file is a matrix of shape n*d, n is dataset size, d
    is feature dimension. Let X be image input, Y be text input, then X[i] and
    Y[i] should be a relevant pair, their concept indicator vector is G[i] from 
    the label file (not required for training)
    example datasets extracted from NUS-WIDE can be download from
    http://www.comp.nus.edu.sg/~wangwei/code/msae

3. write configuration files for sae and MSAE. Examples are included in ./config/.

4. train MSAE
    4.1 train auto-encoders of image sae and text sae layer-wisely
    4.2 train image sae and text sae
    4.3 train msae
    *4.4 visualize training process of msae, python package matplotlib is required

    script run_nuswide.sh provides an example of training, test and visualization on NUS-WIDE dataset


Contact: wangwei@comp.nus.edu.sg
