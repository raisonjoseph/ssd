# SSD
Single Shot Object Detection


## How to setup and run

### Requirements 

- python 3.6 or above
- Following python packages

  `pip install numpy`  
  `pip install opencv-python`  

### Command
`python main.py --prototxt prototxt.txt --model model.caffemodel --image image.jpg --confidence 0.5`
- -p or --prototxt : To define the prototext. Give path to the protot text
- -m or --model : To define moel. GIve path to the model 
- -i or --image : To define image to detect. Give path to the image
- -c or --confidence :  The acceptance criteria that detection required to categorize. Default is 2.o

#### To run as in ths repo
`python main.py --prototxt MobileNetSSD_deploy.prototxt.txt --model ./mobilenet_iter_73000.caffemodel --image car.jpg --confidence 0.3
`
##### Note

There are few images you can tryout for object detection. Have some fun.
