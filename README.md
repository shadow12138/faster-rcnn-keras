# faster-rcnn-keras
faster rcnn based on keras that can train your own dataset

The code is a modification of
https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras
The original code is kind of messy and is based on Jupyter Notebook
I organised it into packages with pycharm.
And I added an extra cnn architecture to it - resnet50

# Available feature extraction cnn architectures:
1.resnet50
2.vgg16

# Train on your own dataset
1.generate annotation file, format as follows:
image_path x1,y1,x2,y2,cls_id x1,y1,x2,y2,cls_id
2.run train.py

# Testing
run test.py

I trained it on a tobacco detection dataset.
The results are shown in the following.

