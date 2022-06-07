# $$TITLE$$
This repository implements <<title>> in PyTorch. Running this code succesfully reproduces the results in the manuscript.
# Training
To train the network on MNIST dataset for 20 epochs on target 0, run the command
```bash 
 python gan_targetted.py --epochs 20 -t 0
```
  The numebr of folds are set to 5 by default. The trained models are saved at the directory ```saved_models/mnist/``` by the name ```D_{target}_{fold} G_{target}_{fold}```.
 
model weights to run ```inference.ipynb``` can be downloaded from the [link](https://www.dropbox.com/sh/nwps3ehuv4rk9dk/AACi84wEPaUHbYs-9xg3ODVOa?dl=0). 
