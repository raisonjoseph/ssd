import numpy as np
import argparse
import cv2
#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
     help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
     help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
     help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
     help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# Calss list of objects we wand to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
     "sofa", "train", "tvmonitor"]

# Varialbe will hold the a unique color for each of the items in class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#Consturct the network based on the model
print("[INFO] loading model…")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Read the image and convert it to the blob format
# Blob is binary format of image. If you wanna know more about it just google it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
print("[INFO] computing object detections…")


# Crucial setup 
# Give the converted blog file to the netwok which will be the one doing the detection
net.setInput(blob)
detections = net.forward()

# For objects detected find out the region and mark it
# We check the confidence range along with the one we entered. Confidence is the acceptable percentage for object to be categorized.
objects = []
for i in np.arange(0, detections.shape[2]):
     confidence = detections[0, 0, i, 2]
     if confidence > args["confidence"]:
          idx = int(detections[0, 0, i, 1])
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          # region where to draw box
          (startX, startY, endX, endY) = box.astype("int")
          area = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)  
          item= {}
          item["startX"] =  startX
          item["startY"] = startY
          item["endX"] =  endX
          item["endY"] = endY
          item["area"] =  area
          item["idx"] = idx
          objects.append(item.copy()) 


print(len(objects))
for object in objects:
     print("[INFO] {}".format(object["area"]))  
	# construct a box containing the image
     cv2.rectangle(image, (object["startX"], object["startY"]), (object["endX"], object['endY']), COLORS[object["idx"]], 2)     
     y = object["startY"] - 15 if object["startY"] - 15 > 15 else object["startY"] + 15     
     cv2.putText(image,  object["area"], (object["startX"]+ 5, y),
     cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[object["idx"]], 2)
cv2.imshow("Output", image)
cv2.waitKey(0)
