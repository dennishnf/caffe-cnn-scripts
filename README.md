
# caffe-cnn-scripts #

Scripts for iterative training with multiple models and datasets.


## a. for a single iteration ##


1. With scripts/generate_lmdb.py generate the "train.txt"/"test.txt", "train_lmdb"/"test_lmdb", "mean_image.binaryproto" files. 


2. Verify models/model-NN/ model_train_val.prototxt model_deploy.prototxt the inputs, parameters of architecture like the size of conv, outputs.


3. Verify in models/model-NN/model_solver.prototxt the model and iterations.


4. Train the network, in terminal:

```
$ cd /home/dennis/Desktop/cnn-caffe-scripts/
```

```
$ caffe train --solver /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/model_solver.prototxt --gpu 0
```

Note: Fine tunning:

```
$ caffe train --solver /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/model_solver.prototxt --weights /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/bvlc_alexnet.caffemodel --gpu 0
```

Note: Resume training:

```
$ caffe train --solver /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/model_solver.prototxt --snapshot /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/train_iter_10000.solverstate
```

5. Test the network, in terminal:

```
$ cd /home/dennis/Desktop/cnn-caffe-scripts/
```

```
$ python /home/dennis/Desktop/cnn-caffe-scripts/scripts/testing_v2.py --proto /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/model_deploy.prototxt --model /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/train_iter_6000.caffemodel --mean /home/dennis/Desktop/cnn-caffe-scripts/input/dataset-MM/mean_image.binaryproto --txt /home/dennis/Desktop/cnn-caffe-scripts/input/dataset-MM/test.txt --cm none
```

```
$ python /home/dennis/Desktop/cnn-caffe-scripts/scripts/testing_v1.py --proto /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/model_deploy.prototxt --model /home/dennis/Desktop/cnn-caffe-scripts/models/model-NN/train_iter_6000.caffemodel --mean /home/dennis/Desktop/cnn-caffe-scripts/input/dataset-MM/mean_image.binaryproto --lmdb /home/dennis/Desktop/cnn-caffe-scripts/input/dataset-MM/test_lmdb
```

6. Draw the model and see the colvolutional filters:

```
cd /home/dennis/Desktop/cnn-caffe-scripts/scripts
```

```
$ python /home/dennis/caffe/python/draw_net.py /home/dennis/Desktop/cnn-caffe-scripts/models/model01/model_train_val.prototxt /home/dennis/Desktop/cnn-caffe-scripts/models/model01/model.png
```

```
$ python visualize_example.py
```


## b. for single and multiple iterations ##


1. With scripts/generate_lmdb.py generate the "train.txt"/"test.txt", "train_lmdb"/"test_lmdb", "mean_image.binaryproto" files. 


2. Verify models/model-NN/ model_train_val.prototxt model_deploy.prototxt the inputs, parameters of architecture like the size of conv, outputs.


3. Verify in models/model-NN/model_solver.prototxt the model and iterations.


4. Then, use scripts/recursive_train.py, this script automatically train, test, draw models, and show convolutional filters.


## License ##

GNU General Public License, version 3 (GPLv3).

You can visit my personal website: [http://dennishnf.com](http://dennishnf.com)

