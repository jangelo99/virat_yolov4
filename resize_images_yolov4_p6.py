
import cv2
import glob
import os.path
import sys

### CONSTANTS ###
INPUT_DIR = "/home/angelojj1/workspace/yolov4/darknet-master/data/virat_ground/validate"
OUTPUT_DIR = "/home/angelojj1/workspace/yolov4/darknet-master/data/virat_ground_p6/validate"
LIST_FILE = "virat_ground_p6_validate.txt"
LIST_PATH = "data/virat_ground_p6/validate/"
# p6 network size is 1280x1280
OUTPUT_SIZE = 1280

list_f = open(LIST_FILE, 'w', newline='\n')

# iterate over all images in input directory
img_list = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
for img_path in img_list:

  img_name = img_path.split('/')[-1]
  print(img_name)
  img = cv2.imread(img_path)
  height, width, _ = img.shape
  bottom_pad = OUTPUT_SIZE - height

  if width == 1920:
    # image is 1920x1080, so split into left and right images
    img_left = img[0:height, 0:OUTPUT_SIZE]
    img_left_pad = cv2.copyMakeBorder(img_left, 0, bottom_pad, 0, 0, cv2.BORDER_CONSTANT)
    out_name = img_name.replace(".jpg", "_left.jpg")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, img_left_pad, [cv2.IMWRITE_JPEG_QUALITY, 100])
    list_f.write(LIST_PATH + out_name + "\n")
    rt_offset = width - OUTPUT_SIZE
    img_rt = img[0:height, rt_offset:]
    img_rt_pad = cv2.copyMakeBorder(img_rt, 0, bottom_pad, 0, 0, cv2.BORDER_CONSTANT)
    out_name = img_name.replace(".jpg", "_right.jpg")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, img_rt_pad, [cv2.IMWRITE_JPEG_QUALITY, 100])
    list_f.write(LIST_PATH + out_name + "\n")

    # now generate labels for the left/right images, as appropriate
    left_f = open(os.path.join(OUTPUT_DIR, img_name.replace(".jpg", "_left.txt")), 'w')
    right_f = open(os.path.join(OUTPUT_DIR, img_name.replace(".jpg", "_right.txt")), 'w')
    label_f = open(img_path.replace(".jpg", ".txt"), 'r')
    for line in label_f:
      coords = line.strip().split(' ')[1:]
      orig_ctr_x = float(coords[0]) * width
      ctr_y = (float(coords[1]) * height) / OUTPUT_SIZE
      box_w = (float(coords[2]) * width) / OUTPUT_SIZE
      box_h = (float(coords[3]) * height) / OUTPUT_SIZE
      if orig_ctr_x < OUTPUT_SIZE:
        # need to write entry to left label file
        ctr_x = orig_ctr_x / OUTPUT_SIZE
        left_f.write('0 %f %f %f %f\n' % (ctr_x, ctr_y, box_w, box_h))
      if orig_ctr_x > rt_offset:
        # need to write entry to right label file
        ctr_x = (orig_ctr_x - rt_offset) / OUTPUT_SIZE
        right_f.write('0 %f %f %f %f\n' % (ctr_x, ctr_y, box_w, box_h))
    
    # close label files
    left_f.close()
    right_f.close()
    label_f.close()

  elif width == 1280:
    # image is 1280x720, so we only need to pad and rescale height
    img_pad = cv2.copyMakeBorder(img, 0, bottom_pad, 0, 0, cv2.BORDER_CONSTANT)
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img_pad, [cv2.IMWRITE_JPEG_QUALITY, 100])
    list_f.write(LIST_PATH + img_name + "\n")
    in_label_f = open(img_path.replace(".jpg", ".txt"), 'r')
    out_label_f = open(out_path.replace(".jpg", ".txt"), 'w')
    for line in in_label_f:
      coords = line.strip().split(' ')[1:]
      ctr_y = (float(coords[1]) * height) / OUTPUT_SIZE
      box_h = (float(coords[3]) * height) / OUTPUT_SIZE
      out_label_f.write('0 %s %f %s %f\n' % (coords[0], ctr_y, coords[2], box_h))
    in_label_f.close()
    out_label_f.close()

  else:
    # shouldn't ever reach this
    print("Unexpected image size for " + img_name)
    sys.exit()

# close list file
list_f.close()
print("Finished!")

