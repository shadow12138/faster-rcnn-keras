# faster-rcnn-keras
faster rcnn based on keras that can train your own dataset

The code is a modification of<br>
https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras<br>
The original code is kind of messy and is based on Jupyter Notebook<br>
I organised it into packages with pycharm.<br>
And I added an extra cnn architecture to it - **resnet50**<br>

# Environment
keras 2.1.6 <br>
tensorflow 1.10.0 - Support both cpu and gpu<br>
Pillow <br>
opencv-python <br>

# Available feature extraction cnn architectures:<br>
1. **resnet50**<br>
2. **vgg16**<br>

# Train on your own dataset
**1**. generate annotation file, format as follows:<br>
image_path1 img_width,img_height x1,y1,x2,y2,cls_id x1,y1,x2,y2,cls_id<br>
image_path2 img_width,img_height x1,y1,x2,y2,cls_id<br>

**2**. download pre-trained weights of vgg or resnet and put it into directory "weights"<br>

vgg - https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5<br>

resnet - https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5<br>

**3**. set your own class_mapping in config.py<br>
for example, if your dataset has three classes<br>
your class_mapping should go like this {'class1': 0, 'class2': 1, 'class3': 2, 'bg':3}<br>

**4**. run train.py<br>

# Testing
run test.py

# Results
i annotated four pokemon pets in about 20 images for training, <br>
they are 皮卡丘(pikachu)/妙蛙种子(bulbasaur)/小火龙(charmander)/杰尼龟(squirtle)  <br>
sorry that i only know four among the pets, also sorry that i call them pets <br>
The result is shown in the following.<br>
![alt text](https://github.com/shadow12138/faster-rcnn-keras/blob/master/results/pokemon_result_00.png)<br>
![alt text](https://github.com/shadow12138/faster-rcnn-keras/blob/master/results/pokemon_result_02.png)<br>
![alt text](https://github.com/shadow12138/faster-rcnn-keras/blob/master/results/pokemon_result_01.png)<br>


I also trained it on a tobacco detection dataset.
The results are shown in the following.<br>
![alt text](https://github.com/shadow12138/faster-rcnn-keras/blob/master/results/tobacco_result.png)<br>

