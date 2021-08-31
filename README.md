# You Only Look Once (YOLO)

# Imports
```python
import cv2
import matplotlib.pyplot as plt
from darknet import Darknet
from utils import detect_objects, load_class_names, plot_boxes, print_objects
```


# Setup the neural network
```python
# Set the location and name of the cfg file
cfg_file = "./cfg/yolov3.cfg"
# Set the location and name of the pre-trained weights file
weight_file = "./weights/yolov3.weights"
# Set the location and name of the COCO object classes file
namesfile = "data/coco.names"
# Load the network architecture
m = Darknet(cfg_file)
# Load the pre-trained weights
m.load_weights(weight_file)
# Load the COCO object classes
class_names = load_class_names(namesfile)
```

# Print the neural network used in YOLOv3

```python
m.print_network()
```

    layer     filters    size              input                output
        0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32
        1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64
        2 conv     32  1 x 1 / 1   208 x 208 x  64   ->   208 x 208 x  32
        3 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64
        4 shortcut 1
        5 conv    128  3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x 128
        6 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64
        7 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128
        8 shortcut 5
        9 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64
       10 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128
       11 shortcut 8
       12 conv    256  3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 256
       13 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
       14 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
       15 shortcut 12
       16 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
       17 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
       18 shortcut 15
       19 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
       20 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
       21 shortcut 18
       22 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
       23 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
       24 shortcut 21
       25 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
       26 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
       27 shortcut 24
       28 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
       29 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
       30 shortcut 27
       31 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
       32 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
       33 shortcut 30
       34 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
       35 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
       36 shortcut 33
       37 conv    512  3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 512
       38 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       39 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       40 shortcut 37
       41 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       42 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       43 shortcut 40
       44 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       45 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       46 shortcut 43
       47 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       48 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       49 shortcut 46
       50 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       51 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       52 shortcut 49
       53 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       54 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       55 shortcut 52
       56 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       57 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       58 shortcut 55
       59 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       60 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       61 shortcut 58
       62 conv   1024  3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x1024
       63 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
       64 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
       65 shortcut 62
       66 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
       67 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
       68 shortcut 65
       69 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
       70 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
       71 shortcut 68
       72 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
       73 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
       74 shortcut 71
       75 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
       76 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
       77 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
       78 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
       79 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
       80 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
       81 conv    255  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 255
       82 detection
       83 route  79
       84 conv    256  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 256
       85 upsample           * 2    13 x  13 x 256   ->    26 x  26 x 256
       86 route  85 61
       87 conv    256  1 x 1 / 1    26 x  26 x 768   ->    26 x  26 x 256
       88 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       89 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       90 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       91 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
       92 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
       93 conv    255  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 255
       94 detection
       95 route  91
       96 conv    128  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 128
       97 upsample           * 2    26 x  26 x 128   ->    52 x  52 x 128
       98 route  97 36
       99 conv    128  1 x 1 / 1    52 x  52 x 384   ->    52 x  52 x 128
      100 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
      101 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
      102 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
      103 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
      104 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
      105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255
      106 detection


# Inference on the images
Images in the images folder
```python
def pred_images(image_name, iou_thresh=0.4, nms_thresh=0.6):
    # first layer of the network is 416 x 416 x 3
    # Set the default figure size
    plt.rcParams["figure.figsize"] = [24.0, 14.0]
    # Load the image
    img = cv2.imread(f"./images/{image_name}.jpg")
    # Convert the image to RGB
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # We resize the image to the input width and height of the first layer of the network.
    resized_image = cv2.resize(original_image, (m.width, m.height))
    # Display the images
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(122)
    plt.title("Resized Image")
    plt.imshow(resized_image)
    plt.show()
    # setting the non maximal suppressin threshold to only keep the best bounding box.
    # set the NMS threshold
    # nms_thresh = 0.9
    # set the intersecion over union threshold
    # After removing all the predicted bounding boxes that have a low detection
    # probability, the second step in NMS, is to select the bounding boxes with
    # the highest detection probability and eliminate all the bounding boxes whose
    # Intersection Over Union (IOU) value is higher than a given IOU threshold.
    # Detect objects in the image
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    # Print the objects found and the confidence level
    print_objects(boxes, class_names)
    # Plot the image with bounding boxes and corresponding object class labels
    plot_boxes(original_image, boxes, class_names, plot_labels=True)
```


```python
pred_images("city_scene", iou_thresh=0.4, nms_thresh=0.6)
```


    
![svg](yolo_files/yolo_4_0.svg)
    


    
    
    It took 1.971 seconds to detect the objects in the image.
    
    Number of Objects Detected: 21 
    
    Objects Found and Confidence Level:
    
    1. person: 0.999996
    2. person: 1.000000
    3. car: 0.707238
    4. truck: 0.666981
    5. person: 1.000000
    6. traffic light: 1.000000
    7. truck: 0.856236
    8. person: 1.000000
    9. person: 1.000000
    10. person: 0.999894
    11. car: 0.666931
    12. person: 0.999990
    13. person: 1.000000
    14. traffic light: 1.000000
    15. traffic light: 1.000000
    16. person: 0.999995
    17. traffic light: 1.000000
    18. traffic light: 0.999999
    19. traffic light: 1.000000
    20. person: 0.999993
    21. person: 0.999996



    
![svg](yolo_files/yolo_4_2.svg)
    

## Acknowledgements

 - [Udacity CVND](https://www.udacity.com/course/computer-vision-nanodegree--nd891)
  