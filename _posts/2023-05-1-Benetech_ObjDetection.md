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


## References
{1} Layout-Parser: https://layout-parser.github.io/

{2} Detectron2: https://github.com/facebookresearch/detectron2
