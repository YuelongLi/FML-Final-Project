# General info
This repository contains some of the scripts from the [AutoAttack repository](https://github.com/fra31/auto-attack).

It also contains some scripts and models from the [TRADES repository](https://github.com/yaodongyu/TRADES).

Some scripts have been copied and modified. We have also copied over the model files from TRADES.

The maxent model used to generate the input data for the SVM (which is implemented by [libsvm](https://github.com/cjlin1/libsvm)) is the ready-to-download CIFAR-10 model from the TRADES repository.

Scripts running AutoAttack are used to generate adversarial data that will then be passed through the TRADES neural network to produce the raw predictions (logits) for each class. This data will then be used as input to the SVM step.

# Dependencies

- Clone [libsvm](https://github.com/cjlin1/libsvm) into the working directory.
- Download the TRADES [CIFAR-10 model](https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view?usp=sharing) and place it in the working directory.
- Run `pip install numpy torch torchvision plotly`

# Instructions

- To generate the plots of the raw predictions of the TRADES neural network, run `plot_raw_preds_from_TRADES.py`. This is the data that is used as input to the SVM provided by `libSVM`. The dimensions that are plotted can be modified.
- To generate the adversarial data, run `gen_adv_data.py`. This generates 2000 training data points and 200 adversarial adversarial data points, all in tensor format.
- To generate the data used in the SVM, run `gen_cat_libsvm_data.py`. This passes the adversarial data through the TRADES neural network and outputs the raw predictions. The data will be separated into data that belongs to class `cat` (label of +1) and data that does not (-1).
- Scale the data and run some hyper parameter tuning for libSVM based on the methods and scripts from homework 2. (See [here](https://github.com/olgavrou/FML-HW2).)
- Train and test the data using libSVM by running `run_svm.sh`, which does the following:
	- `./libsvm/svm-scale -s range libsvm_cat_train.dat > libsvm_cat_train.dat.scale`
	- `./libsvm/svm-scale -s range libsvm_cat_test.dat > libsvm_cat_test.dat.scale`
	- Does some hyper parameter tuning
	- `./libsvm/svm-train -t 1 -d 1 -c 20 -v 5 libsvm_cat_train.dat.scale`
	- `./libsvm/svm-train -t 1 -d 1 -c 20 libsvm_cat_train.dat.scale`
	- `./libsvm/svm-predict libsvm_cat_test.dat.scale libsvm_cat_train.dat.scale.model out`
