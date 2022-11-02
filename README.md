# virat_yolov4

### Python scripts for preparing [VIRAT Video Data](https://viratdata.org/index.html) for training YOLOv4 models:  

**1. `convert_virat_videos.py`** - uses OpenCV to convert VIRAT videos to individual JPEG images

**2. `resize_images_yolov4_p6.py`** - tiles and resizes VIRAT image frames to 1280x1280 pixels for YOLOv4_p6 model training

**3. `virat_to_yolov4_format.py`** - parses VIRAT image frames and annotation files, converts to YOLOv4 training format

   
### Other Python scripts:  

**`prepare_yolov4_data.py`** - converts Caltech Pedestrian dataset in VOC format to YOLOv4 format

**`torchreid_tf2_inference.py`** - uses Tensorflow 2 to perform inference with [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) model
