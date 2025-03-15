# first Resnet


CIFAR-10 Classification Using ResNet

Project Overview

This project implements a Convolutional Neural Network (CNN) based on the ResNet architecture to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes, with 50,000 training images and 10,000 test images.

We use PyTorch to build and train the model, leveraging GPU acceleration for efficient training.

Dataset

 • CIFAR-10 contains 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
 • The dataset is preprocessed using normalization and data augmentation techniques (random crop, horizontal flip) to improve model performance.

Model Architecture

We use a ResNet-18 model, a deep residual network that helps mitigate the vanishing gradient problem through skip connections. The model is adapted for CIFAR-10 by modifying the first convolutional layer and the final fully connected (FC) layer.

Implementation Details

 • Framework: PyTorch
 • Architecture: ResNet-18
 • Optimizer: Adam (learning rate = 0.001)
 • Loss Function: CrossEntropyLoss
 • Batch Size: 128
 • Data Augmentation: Random horizontal flip, normalization
 • Training: 20 epochs on GPU

Results

After training, the model achieves an accuracy of ~85% on the test set. Further improvements can be made by tuning hyperparameters, using data augmentation, or applying learning rate scheduling.

How to Run the Code

 1. Clone the repository:

git clone https://github.com/asleamirhossein/firstResnet.git
cd cifar10-resnet


 2. Install dependencies:

pip install torch torchvision matplotlib  


 3. Run the training script:

python train.py  


 4. Evaluate the model:

python evaluate.py  



Future Improvements

 • Use ResNet-34 or ResNet-50 for better accuracy.
 • Apply learning rate scheduling.
 • Experiment with different optimizers (SGD with momentum, RMSprop).
 • Implement semi-supervised learning for better generalization.


