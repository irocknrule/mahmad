---
title: "Object Detection using Layout Parser and Detectron2"
date: 2023-06-15T15:34:30-04:00
categories:
  - Blog
  - computer vision
tags:
  - kaggle
  - cv
  - image
  - Object Detection
classes: wide
---

Following up on the Kaggle Benetech graphs competition post earlier (https://irocknrule.github.io/mahmad/blog/fastai/computer%20vision/Benetech_image_classifier/) where I described an implementation of a simple but powerful image classifier using fastai, in this post I talk about the next step in the competition where we need to carry out object detection and Optical Character Reader (OCR) from the given images. I had mentioned in the previous post that I selected this particular competition from Kaggle because it required a relatively deep dive into various aspects of Deep Learning (DL) and computer vision, so was ideal avenue to get my feet wet in these areas. 

In this post, we look at how we can train an effective object detection and OCR model using LayoutParser {1} which uses the Detectron2 architecture created by FaceBook Research {2} and is a standard, highly-performant and popular object detection model. 

## Layout Parser and Detectron2

Layout Parser is a well designed document image analyzer which can be used to extract various document structures using only minimal lines of code. It has a set of pre-trained models to extract information from scientific papers, newspapers, documents with tables and so on in the model zoo which have been community driven and can be readily available. In our case for this project, we are trying to extract information from graph images - a type of image which did not have a pre-trained model. Layout Parser does allow training a model from scratch and most of the techniques to train this model is based on Facebook research's Detectron2 object detection libraries. Detectron2 provides numerous object detection and segmentation algorithms and is used significantly within production applications in Meta/Facebook. 

Using these toolsets for this Kaggle task was a great opportunity to learn more about applications of DL in computer vision, something which I have not been exposed to during my regular work at Amazon.

## Graphs Object Detection and OCR

We now take a closer look at the step in training a new model and carrying out OCR for the Benetech graphs competitions. To train and then make inferences from a new model, we have to carry out some initial steps to get the data ready, train the model and then carry out inference on test graph images followed by OCR to read the various text boxes. 

### Input Datasets in COCO format

Data inputs to detectron2 require image annotations with segments/object bounding boxes to be in specific formats and in this case we used the well-known Common Objects in Context (COCO) format {3} .COCO  is a widely used dataset format for computer vision tasks, such as object detection, instance segmentation, and keypoint detection. The COCO dataset provides a standardized format to represent image annotations and their corresponding metadata and it consists of 2 main components: images and annotations. 

- **images:** each image contains information about the image in the dataset such as file name, width, height, a unique ID and some few other optional details. 
- **annotations:** This represents the object instance within an image and contains the ID of the image where it belongs, a specific category ID of the object, the bounding box coordinates of the object in \[x,y,width,height] format and some other attributes representing the object. 

The annotation JSON provided as input for the Kaggle competition was not in this format - we had specific directories for images and annotations and the annotation format had multiple different fields. So we write up a quick python notebook to carry out the transformation from the input to the standard COCO format as shown below:

```
coco_data = {
    "info": {
        "description": "Benetech input images - random 10K sample",
        "version": "1.0",
        "year": 2023,
        "contributor": "",
        "date_created": "2023-06-11"
    },
    "licenses": [],
    "images": [],
    "categories": [],
    "annotations": []
}
```

The code for the converting the 10K random samples from the input dataset of ~60K images is shared in the notebook here: https://github.com/irocknrule/kaggle/blob/main/Bentech-Graphs/convert_input_to_COCO_format.ipynb

The various object categories we are trying to identify are the following:

```python 
coco_data['categories']
[{'id': 1, 'name': 'chart_title'},
 {'id': 2, 'name': 'axis_title'},
 {'id': 3, 'name': 'tick_label'},
 {'id': 4, 'name': 'plot-bb'},
 {'id': 5, 'name': 'x-axis-tick'},
 {'id': 6, 'name': 'y-axis-tick'},
 {'id': 7, 'name': 'other'},
 {'id': 8, 'name': 'tick_grouping'}]
```

### Model Training with Detectron2

Detectron2 has shared some ready start-up scripts to carry out the model training process which we leverage here. Firstly, we need to split the input image and annotations set into training and testing (as required to train all models). A handy ```cocosplit.py``` is used to create a training set from 85% of the input with the remaining set aside of testing/validation. 

```python
python3 cocosplit.py --annotation-path sample_graphs/result.json --split-ratio 0.85 --train sample_graphs/train.json --test sample_graphs/test.json
```

Link to `cocosplit.py` file here: https://github.com/irocknrule/kaggle/blob/main/Bentech-Graphs/cocosplit.py

Now we use the training script provided to start the training process. Note that instead of training with a ipython notebook, I ended up using the script as once it is set up correctly we can pass in relevant parameters and options and let the training process continue. 

``` python
python train_net.py --dataset_name sample_1 --json_annotation_train train.json --image_path_train sample_1/ --json_annotation_val test.json --image_path_val sample_1/ --config-file fast_rcnn_R_50_FPN_3x.yaml --resume 
OUTPUT_DIR . SOLVER.IMS_PER_BATCH 2 SOLVER.MAX_ITER 2000 SOLVER.BASE_LR 0.00025
```

Detectron2 provides a rich range of options for model training and I have been experimenting with some of them to better understand their effects on the model (which I plan on writing about in an upcoming post). For this initial exercise, we keep things simple and pretty much use most options out of the box. The options above are:

1. `--dataset_name:` Detectron2 requires the datasets being used to be 'registered' in the library for various tracking and optimization purposes. So the dataset name is needed to be passed in - for my example here, i ended up using a very generic name (it does not have to be globally unique) but naming these separately would be very useful to keep track.
2. `--json_annotation_{train|test}:` Paths to the training and testing JSON files. 
3. `--image_path_{train|test}`: Paths to the training and testing image files. Note that the script automatically appends 'images/' to the path, so we only include the top level directory here.
4. `--config_file:` A starting training config file, here we re-use the standard RCNN config. Note that this does not use the RCNN model weights as a starting point but only uses the config options. I tried using a standard model such as resnet to start training but the results were unexpected. I will definitely write up my findings from that experiment in an upcoming post. 
5. `--resume:` This is a critical option to pass in as this tells Detectron2 to resume training from the last saved checkpoint instead of starting from scratch. 
6. Training options: Other options above include the total number of training iterations, the base learning rate and the output directory. 

The training script used is: https://github.com/irocknrule/kaggle/blob/main/Bentech-Graphs/train_net.py

### Training results

We carry out training for 2000 iterations which on my Paperspace Gradient P5000 GPU machine takes about ~100 mins. We can of course train for longer, but I was curious to see how this was handled at a relatively short training period. 

For object detection and image segmentation, Average Precision (AP) is used as the primary metric. I will leave the details of this metric out of this post but for those who do not know much about this, I would suggest this highly useful and descriptive post at {4}. 

The training gives us the following results on the test set:

Evaluation results for bbox: 

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |

|:------:|:------:|:------:|:------:|:------:|:------:|

| 38.585 | 68.911 | 36.219 | 34.093 | 21.943 | 42.680 |

Per-category bbox AP: 
| category    | AP     | category      | AP     | category    | AP     |
|:------------|:-------|:--------------|:-------|:------------|:-------|
| chart_title | 41.167 | axis_title    | 49.393 | tick_label  | 63.405 |
| plot-bb     | 74.846 | x-axis-tick   | 35.329 | y-axis-tick | 44.537 |
| other       | 0.000  | tick_grouping | 0.000  |             |        |

The following inferences can be made from the table above:
- The overall AP for all the object categories are 38%
- *tick_label* AP is 63% and for *plot-bb* it is the highest at ~75%.
- The median AP for all objects are ~69%.

## References
{1} Layout-Parser: (https://layout-parser.github.io/)

{2} Detectron2: (https://github.com/facebookresearch/detectron2)

3} COCO format: (https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html)

{4}What is Average Precision in Object Detection & Localization Algorithms and how to calculate it? : https://towardsdatascience.com/what-is-average-precision-in-object-detection-localization-algorithms-and-how-to-calculate-it-3f330efe697b
