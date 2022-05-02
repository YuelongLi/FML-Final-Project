# General info
This repository contains some of the scripts from auto-attack repository: https://github.com/fra31/auto-attack
It also contains some scripts/models form the TRADES repository: https://github.com/yaodongyu/TRADES

Some scripts have been copied and modified. We have also copied over the model files from TRADES.

The model used to generate the input data for the SVM part (using libsvm) is the CIFAR-10 model from the TRADES repository and it can be downloaded.

Auto-attack scripts are used to generate adversarial data that will then be passed through the TRADES neural network to produce the raw predictions (logits) for each class. This data will then be used as input to the libsvm step.

# Dependencies

- clone libsvm in the work directory: https://github.com/cjlin1/libsvm
- download the CIFAR-10 model and place it in the work directory [download link](https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view?usp=sharing)
- pip install numpy torch torchvision plotly

# Instructions

- To generate the plots of the raw predictions of the TRADES nn, which are the data used as input to the libSVM step, run `plot_raw_preds_from_TRADES.py`. You can play around with the dimensions that you wish to plot
- To generate the adversarial data run `gen_adv_data.py` which will generate 2000 training and 200 test data adversarial data points in tensor format
- To generate the data used in the libSVM step run `gen_cat_libsvm_data.py`. This will take the adversarial data, pass them through the loaded TRADES neural network and get back the raw predictions. It will use the label to separate the data into CAT (+1) and non-CAT labels (-1). 
- We then scaled and ran some hyper parameter tuning for libSVM based on the method and scripts from homework 2 see [here](https://github.com/olgavrou/FML-HW2)
- Then we trained and tested the test data using libSVM:
 - ./libsvm/svm-scale -s range libsvm_cat_train.dat > libsvm_cat_train.dat.scale
 - ./libsvm/svm-scale -s range libsvm_cat_test.dat > libsvm_cat_test.dat.scale
 - do some hyper parameter tuning
 - ./libsvm/svm-train -t 1 -d 1 -c 20 -v 5 libsvm_cat_train.dat.scale
 - ./libsvm/svm-train -t 1 -d 1 -c 20 libsvm_cat_train.dat.scale
 - ./libsvm/svm-predict libsvm_cat_test.dat.scale libsvm_cat_train.dat.scale.model out
