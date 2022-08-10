# DL-CapsNet
This is the code for Deep and Light Capsule Network (DL-CapsNet) paper published in DASIP2022.

The following codes are used for this project: <br />
https://github.com/XifengGuo/CapsNet-Keras
https://github.com/brjathu/deepcaps


# Installation
`conda install -c anaconda tensorflow-gpu=1.13.1`
`conda install -c anaconda keras-gpu`
`conda install -c anaconda scipy=1.2*`
`conda install -c conda-forge matplotlib`
`conda install -c conda-forge pillow`

# Usage
The "Main.py" file trains the network and prints the results to the files in the specified folder (input args). <br />
Parameters:<br />
`--dset`: Choice of dataset (options: MNIST, F-MNIST, SVHN, CIFAR-10, CIFAR-100 and affNIST)<br />
`--bsize`: Batch size<br />
`--nepoch`: Number of epochs to train the model<br />
`--drp_rate`: The rate for the capsule dropout <br />
`--res_folder`: The output folder to print the results into<br />
