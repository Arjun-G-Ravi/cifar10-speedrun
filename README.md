# cifar10-speedrun
This repository is a my attempt at cifar10 speedrun challenge.

### Dataset
CIFAR-10 is a foundational dataset in machine learning and computer vision, created in 2009 by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton at the University of Toronto. It consists of 60,000 32x32 color images divided into 10 mutually exclusive classes (e.g., airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 50,000 training images and 10,000 test images. Each class has exactly 6,000 images, making it balanced but challenging due to low resolution, variability in poses/lighting, and inter-class similarities
- Human accuracy on CIFAR-10 is around 94% (per Andrej Karpathy's manual labeling experiment), so models exceeding this are superhuman.

### Challenge
- The goal is to reduce the training time of a model to get the accuracy of 94% and 96% on the CIFAR-10 dataset in one A100 GPU at 400W power.
- End-to-end time: Includes data loading, forward/backward passes, but excludes dataset download.
- No pre-training or external data; pure from-scratch training on CIFAR-10.
- Motivations: Accelerate research (faster experiments), discover new training phenomena (e.g., optimizers, augmentations), and push hardware/software limits. As one researcher notes, it's like a "telescope" for neural network behavior, revealing insights that may generalize to larger models like LLMs.

# Important 
Current record by Keller Jordan
    - 2.59s (94%)
    - 27.3s (96%)
- hlb-CIFAR10: https://github.com/tysam-code/hlb-CIFAR10
- airbench -  https://github.com/KellerJordan/cifar10-airbench

