import cv2
import glob
import os
import os.path
import shutil
import yaml

ANNO_DIR = r"C:\VBox_Share\Data\VIRAT\VIRAT_Ground\annotations_r1\viratannotations-master\train"
IMAGES_DIR = r"C:\VBox_Share\Data\VIRAT\VIRAT_Ground\images_r1"
VIDEOS_DIR = r"C:\VBox_Share\Data\VIRAT\VIRAT_Ground\videos_r1\train"

OUTPUT_DIR = r"C:\VBox_Share\Data\VIRAT\VIRAT_Ground\yolov4\train"
OUTPUT_FILE = "virat_ground_train.txt"

list_f = open(OUTPUT_FILE, 'w', newline='\n')

# iterate over all *.mp4 files in VIDEOS_DIR...

# video_list = glob.glob(os.path.join(VIDEOS_DIR, '*.mp4'))
# for fname in video_list:

fname = r"C:\VBox_Share\Data\VIRAT\VIRAT_Ground\videos_r1\train\VIRAT_S_000201_03_000640_000672.mp4"

# get video height and width
video_name = fname.split('\\')[-1].replace('.mp4', '')
vidcap = cv2.VideoCapture(fname)
width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
del vidcap

# check for annotations files
geom_file = os.path.join(ANNO_DIR, video_name + '.geom.yml')
types_file = os.path.join(ANNO_DIR, video_name + '.types.yml')
if os.path.isfile(geom_file) and os.path.isfile(types_file):

   # get list of person IDs from types file
   id_list = list()
   with open(types_file, 'r') as f:
      yaml_doc = yaml.safe_load(f)
      for block in yaml_doc:
         if 'types' in block and 'Person' in block['types']['cset3']:
            id_list.append(block['types']['id1'])

   # parse geom file to get yolov4 annotations
   anno_dict = dict()
   with open(geom_file, 'r') as f:
      yaml_doc = yaml.safe_load(f)
      for block in yaml_doc:
         if 'geom' in block and block['geom']['id1'] in id_list:
            frame_num = block['geom']['ts0']
            if not frame_num in anno_dict:
               anno_dict[frame_num] = []
            anno_dict[frame_num].append(block['geom']['g0'])

   # iterate over annotations dict and write out files for yolov4...
   for frame_num in anno_dict.keys():
      # check for existence of image file
      img_name = video_name + "_frame_" + str(frame_num) + ".jpg"
      img_file = os.path.join(IMAGES_DIR, video_name, img_name)
      if os.path.isfile(img_file):
         # copy image to output dir and create *.txt file
         shutil.copyfile(img_file, os.path.join(OUTPUT_DIR, img_name))
         list_f.write("data/virat_ground/train/" + img_name + "\n")
         out_file = os.path.join(OUTPUT_DIR, img_name.replace('.jpg', '.txt'))
         with open(out_file, 'w', newline='\n') as f:
            for geom in anno_dict[frame_num]:
               bbox = geom.split(' ')
               xmin, ymin, xmax, ymax  = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
               x_ctr = ((xmax + xmin) / 2.0) / width
               y_ctr = ((ymax + ymin) / 2.0) / height
               bbox_width = (xmax - xmin) / width
               bbox_height = (ymax - ymin) / height
               f.write('0 %f %f %f %f\n' % (x_ctr, y_ctr, bbox_width, bbox_height))

# cleanup
list_f.close()
