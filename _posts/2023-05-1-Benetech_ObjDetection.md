Following up on the Kaggle Benetech graphs competition post earlier (https://irocknrule.github.io/mahmad/blog/fastai/computer%20vision/Benetech_image_classifier/) where I described an implementation of a simple but powerful image classifier using fastai, in this post I talk about the next step in the competition where we need to carry out object detection and Optical Character Reader (OCR) from the given images. I had mentioned in the previous post that I selected this particular competition from Kaggle because it required a relatively deep dive into various aspects of Deep Learning (DL) and computer vision, so was ideal avenue to get my feet wet in these areas. 

In this post, we look at how we can train an effective object detection and OCR model using LayoutParser {1} which uses the Detectron2 architecture created by FaceBook Research {2} and is a standard, highly-performant and popular object detection model. 

## Layout Parser and Detectron2

Layout Parser is a well designed document image analyzer which can be used to extract various document structures using only minimal lines of code. It has a set of pre-trained models to extract information from scientific papers, newspapers, documents with tables and so on in the model zoo which have been community driven and can be readily available. In our case for this project, we are trying to extract information from graph images - a type of image which did not have a pre-trained model. Layout Parser does allow training a model from scratch and most of the techniques to train this model is based on Facebook research's Detectron2 object detection libraries. Detectron2 provides numerous object detection and segmentation algorithms and is used significantly within production applications in Meta/Facebook. 

Using these toolsets for this Kaggle task was a great opportunity to learn more about applications of DL in computer vision, something which I have not been exposed to during my regular work at Amazon.

## Graphs Object Detection and OCR

We now take a closer look at the step in training a new model and carrying out OCR for the Benetech graphs competitions. To train and then make inferences from a new model, we have to carry out some initial steps to get the data ready, train the model and then carry out inference on test graph images followed by OCR to read the various text boxes. 

### Input Datasets in COCO format




## References
{1} Layout-Parser: https://layout-parser.github.io/
{2} Detectron2: https://github.com/facebookresearch/detectron2
