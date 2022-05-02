#!/bin/bash

./libsvm/svm-scale -s range libsvm_cat_train.dat > libsvm_cat_train.dat.scale 
./libsvm/svm-scale -s range libsvm_cat_test.dat > libsvm_cat_test.dat.scale

# run some hyper parameter tuning with libsvm using the same scripts and logic from homework 2 (https://github.com/olgavrou/FML-HW2)

./libsvm/svm-train -t 1 -d 1 -c 20 -v 5 libsvm_cat_train.dat.scale
./libsvm/svm-train -t 1 -d 1 -c 20 libsvm_cat_train.dat.scale
./libsvm/svm-predict libsvm_cat_test.dat.scale libsvm_cat_train.dat.scale.model out