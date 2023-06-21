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

Following up on the Kaggle Benetech graphs competition post earlier [(Simple Image classification using fastai)](https://irocknrule.github.io/mahmad/blog/fastai/computer%20vision/Benetech_image_classifier/) where I described an implementation of a simple but powerful image classifier using fastai, in this post I talk about the next step in the competition where we need to carry out object detection and Optical Character Reader (OCR) from the given images. I had mentioned in the previous post that I selected this particular competition from Kaggle because it required a relatively deep dive into various aspects of Deep Learning (DL) and computer vision, so was ideal avenue to get my feet wet in these areas. 

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

```python
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
|:-----------:|:------:|:-------------:|:------:|:-----------:|:------:|
| chart_title | 41.167 | axis_title    | 49.393 | tick_label  | 63.405 |
| plot-bb     | 74.846 | x-axis-tick   | 35.329 | y-axis-tick | 44.537 |
| other       | 0.000  | tick_grouping | 0.000  |             |        |

The following inferences can be made from the table above:
- The overall AP for all the object categories are 38%
- *tick_label* AP is 63% and for *plot-bb* it is the highest at ~75%.
- The median AP for all objects are ~69%.

## Inference
We now have a trained object detection model and use LayoutParser to first generate bounding boxes for all the elements in the graph and then follow it up by OCR to read the values of these boxes. We will delve into more detail with OCR and its results in a follow up post, as we need to train the model for a longer period of time to have effective results. So here we will touch on the simpler inference from the trained model. 

We create a LayoutParser detectron model and provide the path to the config and final model files along-with the label map of the classes we are trying to predict. We also provide a score threshold as a confidence bound to let the model know how sure we would like it to be before making a prediction. In the example below, again due to low model training time I set the threshold to 0.5. 

```python
import requests
import layoutparser as lp
import cv2
from pycocotools.coco import COCO

model = lp.Detectron2LayoutModel(
    config_path = "config.yaml",
    model_path = "model_final.pth",
    extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.50],
                   label_map={0: "chart_title", 1:"axis_title", 2:"tick_label", 3:"plot-bb",4:"x-axis-tick", 5:"y-axis-tick", 6:"other", 7:"tick_grouping"} # <-- Only output high accuracy preds
)
```

We then read an image from the test set (this image was originally part of the training set shared by the organizers but since we selected 10K images for training, I re-purposed more images from the training set into the test set. Suffice to say these test images were not seen by the model during training.)

```python
image = cv2.imread("../data/test/images/45c20af1f6b8.jpg")
image = image[..., ::-1]

layout = model.detect(image)
lp.draw_box(image, layout, box_width=3)
```

This gives us the following object with bounding boxes. 

{% include figure image_path="/assets/images/blogs/graph+allbbox+0.5thresh.png" alt="" caption="LayoutParser bounding boxes from model inference based on confidence threshold of 0.5"%}

We can see that the model is doing a relatively good job of identifying the chart title, plot bounding box, axis titles and the x and y tick labels. If we go back to the evaluation results, we observe that the AP for the plot bounding box was highest at 74 while tick labels were relatively high at 63. The axis titles had an AP of about 50% too and since we specified a threshold of 50, the x and y tick values were not plotted. 

Let us also look at an instance of using LayoutParser's OCR engine to extract some information from the graph. An interesting use case is the chart title. Lets first plot only the *chart_title* bounding box:

```python
graphtitle_blocks = lp.Layout([b for b in layout if b.type=='chart_title'])
lp.draw_box(image, graphtitle_blocks,box_width=3,show_element_id=True)
```

{% include figure image_path="/assets/images/blogs/graph+charttitle+0.5thresh.png" alt="" caption="Chart Title bounding boxes as inferred from the model."%}

We see that LayoutParser assigns 2 elements to the chart title based on the confidence threshold. In any case, lets go ahead and extract the text using OCR. 

```python
ocr_agent = lp.TesseractAgent(languages='eng')
or block in graphtitle_blocks:
    segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))
        # add padding in each image segment can help
        # improve robustness

    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)
    
for txt in graphtitle_blocks.get_texts():
    print(txt, end='\n---\n')
```

The authors of LayoutParser suggest adding a padding across the bounding box of the OCR text to be read to enable more accurate OCR extraction, so the code above first does that before cropping that part of the image before sending it to the Tesseract OCR engine. We see that the end result is accurate (if we consider both blocks of the chart title).

```
Deaths - Malaria - Sex: Both - Age: 50-69 years (Number) in Malaysia

---
Deaths - Malaria - Sex: Both - Age:

---
```

## Wrapping up

In this post we trained an object detection model from scratch using Detectron2 which is also used by the handy LayoutParser library to detect layouts from images. We had to convert the provide annotations into COCO format and then trained a Detectron2 model followed by an inference example. In the next post, I will dive deeper into the OCR extraction using LayoutParser for this Kaggle competition and put things together for the entire workflow to round out this really interesting project.

## References
{1} Layout-Parser: [https://layout-parser.github.io/](https://layout-parser.github.io/)

{2} Detectron2: [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

3} COCO format: [https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html)

{4}What is Average Precision in Object Detection & Localization Algorithms and how to calculate it? : [https://towardsdatascience.com/what-is-average-precision-in-object-detection-localization-algorithms-and-how-to-calculate-it-3f330efe697b](https://towardsdatascience.com/what-is-average-precision-in-object-detection-localization-algorithms-and-how-to-calculate-it-3f330efe697b)
