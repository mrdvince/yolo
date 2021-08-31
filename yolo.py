# %%
import cv2
import matplotlib.pyplot as plt

from darknet import Darknet
from utils import detect_objects, load_class_names, plot_boxes, print_objects

# %%
# setup the neural network
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
# %%
# Print the neural network used in YOLOv3
m.print_network()
# %%


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


# %%
pred_images("city_scene", iou_thresh=0.4, nms_thresh=0.6)
