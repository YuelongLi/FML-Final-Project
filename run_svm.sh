#!/bin/bash

named_classes=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck")
for i in $(seq 0 9); do
    class_name=${named_classes[i]}

    ./libsvm/svm-scale -s range_"$class_name" libsvm_"$class_name"_train.dat > libsvm_"$class_name"_train.dat.scale 
    ./libsvm/svm-scale -s range_"$class_name" libsvm_"$class_name"_test.dat > libsvm_"$class_name"_test.dat.scale

    # ./libsvm/svm-train -t 1 -d 1 -c 20 -v 5 libsvm_"$class_name"_train.dat.scale
    ./libsvm/svm-train -t 1 -d 1 -c 20 libsvm_"$class_name"_train.dat.scale
    ./libsvm/svm-predict libsvm_"$class_name"_test.dat.scale libsvm_"$class_name"_train.dat.scale.model out_"$class_name"
done