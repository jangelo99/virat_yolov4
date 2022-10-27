import cv2
import glob
import os
import os.path

INPUT_DIR = r"C:\VBox_Share\Data\VIRAT\VIRAT_Ground\videos_r1\train"
OUTPUT_DIR = r"C:\VBox_Share\Data\VIRAT\VIRAT_Ground\images_r1"

ANNO_DIR = r"C:\VBox_Share\Data\VIRAT\VIRAT_Ground\annotations_r1\viratannotations-master\train"

# iterate over all *.mp4 files in input directory
video_list = glob.glob(os.path.join(INPUT_DIR, '*.mp4'))
for fname in video_list:

   # check for annotations files
   video_name = fname.split('\\')[-1].replace('.mp4', '')
   geom_file = os.path.join(ANNO_DIR, video_name + '.geom.yml')
   types_file = os.path.join(ANNO_DIR, video_name + '.types.yml')
   
   if os.path.isfile(geom_file) and os.path.isfile(types_file):
   
      print("Extracting images for " + video_name + "...")

      # create output directory for images and extract video frames
      out_path = os.path.join(OUTPUT_DIR, video_name)
      os.mkdir(out_path)
      count = 0
      vidcap = cv2.VideoCapture(fname)
      success, img = vidcap.read()

      while success:
         img_path = os.path.join(out_path, "%s_frame_%d.jpg" % (video_name, count))
         cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
         count += 1
         success, img = vidcap.read()

   else:
      print("Skipping " + video_name + "!")
      
print("Finished!")