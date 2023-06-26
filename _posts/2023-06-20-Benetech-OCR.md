---
title: "Benetech - OCR extraction with Tesseract and LayoutParser"
date: 2023-06-20T15:34:30-04:00
categories:
  - Blog
  - computer vision
tags:
  - kaggle
  - cv
  - image
  - Object Detection
  - OCR
classes: wide
---

Following up on the previous series of posts for the **Benetech - Making Graphs Accessible** Kaggle competition, this post puts the final steps together using OCR to extract the various graph elements which can then be used to make the final submission. A point to note here is that my aim here was not to actually come up with a final Kaggle submission (i would have definitely liked to submit it, but I overshot timelines and so was not able to submit my final result) but use this very interesting project to learn various aspects of computer vision and apply the learnt concepts here. Moreover, with a ready dataset this competition helped expose me to all the various domains of image classification, object detection, OCR and a sidebar on to GANs as well. 

## Training the model longer

My previous post [here](https://irocknrule.github.io/mahmad/blog/computer%20vision/Benetech_ObjDetection/) describes training and inference of the graph images using Detectron2 and COCO annotations. With a quick and dirty testing solution, we trained the model for only about 2000 iterations and obtained various Average Precision (AP) metrics on the graph elements (the values are listed in the table on the post). I went back to the model and let the training continue up-to 45000 iterations to get a model with better AP values as listed below. 

| category    | AP     | category      | AP     | category    | AP     |  
|:------------|:-------|:--------------|:-------|:------------|:-------|  
| chart_title | 53.491 | axis_title    | 56.646 | tick_label  | 65.561 |  
| plot-bb     | 81.177 | x-axis-tick   | 37.419 | y-axis-tick | 45.830 |  
| other       | 0.000  | tick_grouping | 0.000  |             |        |

We can see that the AP for the chart title has improved to 53% (from 46%) and the *plot-bb* bounding box prediction has now gone up to 81%. 

## Inference

We load the model and carry out inference on a new image and plot the inferred graph elements:

{% include figure image_path="/assets/images/blogs/graps+all+elements+predictions.png" alt="" caption="LayoutParser bounding boxes from model inference based on confidence threshold of 0.5"%}

The overall layout object contains each element predicted by the model. Going through the layout below, we can see every graph element we train (in the *type* field), their bounding boxes and the confidence scores. 

```
Layout(_blocks=[TextBlock(block=Rectangle(x_1=83.58659362792969, y_1=19.001148223876953, x_2=512.0, y_2=339.81005859375), text=None, id=None, type=plot-bb, parent=None, next=None, score=0.9999285936355591), TextBlock(block=Rectangle(x_1=109.52410888671875, y_1=332.4011535644531, x_2=136.0870819091797, y_2=348.6235046386719), text=None, id=None, type=tick_label, parent=None, next=None, score=0.9879248142242432), TextBlock(block=Rectangle(x_1=163.1214599609375, y_1=332.29058837890625, x_2=189.50341796875, y_2=348.1521911621094), text=None, id=None, type=tick_label, parent=None, next=None, score=0.9862905144691467), TextBlock(block=Rectangle(x_1=216.4418182373047, y_1=331.8377380371094, x_2=243.3680877685547, y_2=348.44561767578125), text=None, id=None, type=tick_label, parent=None, next=None, score=0.9856029748916626), TextBlock(block=Rectangle(x_1=374.8288269042969, y_1=332.3216857910156, x_2=400.55902099609375, y_2=348.542724609375), text=None, id=None, type=tick_label, parent=None, next=None, score=0.9854061603546143), TextBlock(block=Rectangle(x_1=31.652496337890625, y_1=197.54367065429688, x_2=112.14673614501953, y_2=216.70425415039062), text=None, id=None, type=tick_label, parent=None, next=None, score=0.9831041097640991), TextBlock(block=Rectangle(x_1=428.7196960449219, y_1=332.5783996582031, x_2=453.2532653808594, y_2=347.89056396484375), text=None, id=None, type=tick_label, parent=None, next=None, score=0.977637529373169), TextBlock(block=Rectangle(x_1=10.71838092803955, y_1=108.47168731689453, x_2=30.42671775817871, y_2=264.4534912109375), text=None, id=None, type=axis_title, parent=None, next=None, score=0.9726553559303284), TextBlock(block=Rectangle(x_1=324.1600341796875, y_1=332.1687927246094, x_2=349.5140686035156, y_2=348.18365478515625), text=None, id=None, type=tick_label, parent=None, next=None, score=0.9677556753158569), TextBlock(block=Rectangle(x_1=481.096923828125, y_1=332.3185729980469, x_2=506.3445129394531, y_2=349.5018615722656), text=None, id=None, type=tick_label, parent=None, next=None, score=0.9480324387550354), TextBlock(block=Rectangle(x_1=34.6596565246582, y_1=82.61080932617188, x_2=111.8458480834961, y_2=101.35513305664062), text=None, id=None, type=tick_label, parent=None, next=None, score=0.939801812171936), TextBlock(block=Rectangle(x_1=462.48162841796875, y_1=358.6455383300781, x_2=511.6942138671875, y_2=380.3831481933594), text=None, id=None, type=axis_title, parent=None, next=None, score=0.9347937107086182), TextBlock(block=Rectangle(x_1=490.7869873046875, y_1=318.7591857910156, x_2=500.4998779296875, y_2=328.09881591796875), text=None, id=None, type=x-axis-tick, parent=None, next=None, score=0.9248719215393066), TextBlock(block=Rectangle(x_1=271.5494079589844, y_1=332.1036071777344, x_2=296.1694030761719, y_2=348.375244140625), text=None, id=None, type=tick_label, parent=None, next=None, score=0.9169681668281555), TextBlock(block=Rectangle(x_1=116.80482482910156, y_1=318.7407531738281, x_2=127.02873229980469, y_2=328.5643005371094), text=None, id=None, type=x-axis-tick, parent=None, next=None, score=0.8989273905754089), TextBlock(block=Rectangle(x_1=287.4098815917969, y_1=360.9184875488281, x_2=329.6842956542969, y_2=379.5876159667969), text=None, id=None, type=axis_title, parent=None, next=None, score=0.8695796728134155), TextBlock(block=Rectangle(x_1=117.57305145263672, y_1=202.40631103515625, x_2=127.0783462524414, y_2=211.4820098876953), text=None, id=None, type=y-axis-tick, parent=None, next=None, score=0.8665221929550171), TextBlock(block=Rectangle(x_1=437.04449462890625, y_1=27.74579620361328, x_2=446.4718017578125, y_2=36.57726287841797), text=None, id=None, type=y-axis-tick, parent=None, next=None, score=0.8549107313156128), TextBlock(block=Rectangle(x_1=34.67308807373047, y_1=312.3880310058594, x_2=115.93985748291016, y_2=332.3302001953125), text=None, id=None, type=tick_label, parent=None, next=None, score=0.8230078816413879), TextBlock(block=Rectangle(x_1=494.1811828613281, y_1=332.26025390625, x_2=507.98345947265625, y_2=348.24676513671875), text=None, id=None, type=tick_label, parent=None, next=None, score=0.7852360010147095), TextBlock(block=Rectangle(x_1=185.66152954101562, y_1=0.5711309909820557, x_2=438.5762939453125, y_2=24.00111961364746), text=None, id=None, type=axis_title, parent=None, next=None, score=0.6850646734237671), TextBlock(block=Rectangle(x_1=438.1932067871094, y_1=318.5933532714844, x_2=447.7580261230469, y_2=327.7224426269531), text=None, id=None, type=y-axis-tick, parent=None, next=None, score=0.6137822270393372), TextBlock(block=Rectangle(x_1=277.4983825683594, y_1=318.444091796875, x_2=287.4945068359375, y_2=328.1589050292969), text=None, id=None, type=x-axis-tick, parent=None, next=None, score=0.575972855091095), TextBlock(block=Rectangle(x_1=224.36740112304688, y_1=318.2536926269531, x_2=234.5363006591797, y_2=328.139892578125), text=None, id=None, type=y-axis-tick, parent=None, next=None, score=0.5637820363044739), TextBlock(block=Rectangle(x_1=440.5396728515625, y_1=332.323974609375, x_2=454.738525390625, y_2=347.50811767578125), text=None, id=None, type=tick_label, parent=None, next=None, score=0.5404101014137268), TextBlock(block=Rectangle(x_1=278.4656677246094, y_1=318.32501220703125, x_2=288.0555725097656, y_2=327.6543273925781), text=None, id=None, type=y-axis-tick, parent=None, next=None, score=0.5336461663246155), TextBlock(block=Rectangle(x_1=154.67581176757812, y_1=0.0, x_2=447.9066467285156, y_2=27.360971450805664), text=None, id=None, type=chart_title, parent=None, next=None, score=0.5228949785232544)], page_data={})
```

Similar to the previous post, we can now plot and extract individual values within the element via OCR. For example below, we pad the element to be large enough and all the elements of the graph are then extracted here - the title, the x and y-axis lables, axis titles etc.

{% include figure image_path="/assets/images/blogs/graps+plot-bb+predictions.png" alt="" caption="Plot bounding box as predicted by the model."%}

```python
get_ocr_text(element_blocks, image, left=25, right=25, top=25, bottom=25)
```

Output:

```
13284

13276

13268

0.0

Money supply rule: Unstable case

0.4

08

1.6

2.0

24

 

28

---
```

We extract the chart title by:

```python
element_blocks = plot_elements('chart_title', layout, image)
lp.draw_box(image, element_blocks,box_width=3, show_element_id=True)

```

{% include figure image_path="/assets/images/blogs/graps+chart-title+predictions.png" alt="" caption="Plot chart title as predicted by the model."%}{: width="100"}


```python
get_ocr_text(element_blocks, image)
```

```
Money supply rule: Unstable case
```

We can use the same techniques to extract other graph elements and save them to be used in any other way. 

The overall notebook is present here: [https://github.com/irocknrule/kaggle/blob/main/Bentech-Graphs/inference_graphs.ipynb](https://github.com/irocknrule/kaggle/blob/main/Bentech-Graphs/inference_graphs.ipynb)

## Wrapping-up

Overall, this particular project/competition in Kaggle was extremely interesting as we worked on so many different aspects to help solve the problem. We started off using *fastai* for image classification, detectron2 for object detection, COCO JSON annotations to help carry out model training for Detectron2 and then finally LayoutParser and its OCR engine to extract text. I hope to be able to apply each of these tools and concepts in more depth in other problems in the upcoming future as this provided an ideal start in all these domains. 
