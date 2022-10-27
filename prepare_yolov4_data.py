
import glob
import xml.etree.ElementTree as ET

INPUT_DIR = r"C:\VBox_Share\Data\CalTech_Pedestrian\data\voc_output"
OUTPUT_DIR = r"C:\VBox_Share\Data\CalTech_Pedestrian\data\yolov4\obj"

TRAINING_SETS = ['set00', 'set01', 'set03', 'set04', 'set05', 'set07']
TESTING_SETS = ['set08', 'set09', 'set10']

TRAIN_FILE = "caltech_train.txt"
VALID_FILE = "caltech_valid.txt"

train_f = open(TRAIN_FILE, 'w', newline='\n')
valid_f = open(VALID_FILE, 'w', newline='\n')

glob_pattern = INPUT_DIR + '\\annotations\\**\\*.xml'
file_list = glob.glob(glob_pattern, recursive='True')

count = 0

for file_path in file_list:
   # determine if file in training set
   file_name = file_path.split("\\")[-1]
   set_str = file_name[0:5]
   is_training = set_str in TRAINING_SETS

   # get image size info
   root = ET.parse(file_path).getroot()
   size_elem = root.find('size')
   width = float(size_elem.find('width').text)
   height = float(size_elem.find('height').text)

   # create corresponding *.txt file for yolov4
   out_file = OUTPUT_DIR + '\\' + file_name.replace('.xml', '.txt')
   with open(out_file, 'w', newline='\n') as f:
      for obj_elem in root.findall('object'):
         bbox_elem = obj_elem.find('bndbox')
         bbox_xmin = float(bbox_elem.find('xmin').text)
         bbox_xmax = float(bbox_elem.find('xmax').text)
         bbox_ymin = float(bbox_elem.find('ymin').text)
         bbox_ymax = float(bbox_elem.find('ymax').text)
         x_ctr = ((bbox_xmax + bbox_xmin) / 2.0) / width
         y_ctr = ((bbox_ymax + bbox_ymin) / 2.0) / height
         bbox_width = (bbox_xmax - bbox_xmin) / width
         bbox_height = (bbox_ymax - bbox_ymin) / height
         f.write('0 %f %f %f %f\n' % (x_ctr, y_ctr, bbox_width, bbox_height))

   # write entry to either train or valid list file
   out_str = 'data/obj/' + file_name.replace('.xml', '.jpg') + '\n'
   if is_training:
      train_f.write(out_str)
   else:
      valid_f.write(out_str)

   count += 1
   if (count % 1000) == 0:
      print(str(count))

# cleanup
train_f.close()
valid_f.close()