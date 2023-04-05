**Accepted at GCV(https://generative-vision.github.io/workshop-CVPR-23/), CVPR23**

This repository implements [Explaining Adversarial Attacks From a
Geometric Perspective](https://www.google.com) in PyTorch. Running this code succesfully reproduces the results in the manuscript.
# Training
To train the network on MNIST dataset for 20 epochs on target 0, run the command
```bash 
 python gan_adv.py --epochs 20 -t 0
```
  The numebr of folds are set to 5 by default. The trained models are saved at the directory ```saved_models/mnist/``` by the name ```D_{target}_{fold} G_{target}_{fold}```.
 
Please run `python gan_adv.py --help` for details of the possible arguments to pass to the `gan_adv.py` script.
 
 
Model weights to run ```inference.ipynb``` can be downloaded from the [link](https://www.dropbox.com/sh/nwps3ehuv4rk9dk/AACi84wEPaUHbYs-9xg3ODVOa?dl=0). 
