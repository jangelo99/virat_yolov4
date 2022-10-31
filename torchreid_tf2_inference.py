import glob
import numpy as np
import tensorflow as tf

from numpy.linalg import norm
from PIL import Image

saved_model_dir = "models/osnet_ain_x1_0_ms_d_c_tf"
img_path = "./data/cville_test/gallery/*.jpg"
img_file_list = glob.glob(img_path)

# for RGB image normalization
norm_mean = [0.485, 0.456, 0.406] # imagenet mean
norm_std = [0.229, 0.224, 0.225] # imagenet std

# Query image from cam 0
img0 = Image.open('./data/cville_test/query/DJI_0009_pid_0_cam_0.jpg')
x0 = tf.keras.preprocessing.image.img_to_array(img0)
# normalize
x0 /= 255.0
for i in range(3):
   x0[:, :, i] = (x0[:, :, i] - norm_mean[i]) / norm_std[i]
x0 = np.expand_dims(x0, axis=0)
x0 = np.moveaxis(x0, -1, 1)

# Load gallery images from "cam 1"
x_list = []
for img_file in img_file_list:
   img = Image.open(img_file)
   x = tf.keras.preprocessing.image.img_to_array(img)
   x /= 255.0
   for i in range(3):
      x[:, :, i] = (x[:, :, i] - norm_mean[i]) / norm_std[i]
   x = np.expand_dims(x, axis=0)
   x = np.moveaxis(x, -1, 1)
   x_list.append(x)

# Load torchreid model
model = tf.saved_model.load(saved_model_dir)
sig_def = model.signatures["serving_default"]
y0 = sig_def(images=x0)['output'].numpy()[0]

# Run inference and calculate cos similarity values
for i, x1 in enumerate(x_list):
   print(img_file_list[i])
   y1 = sig_def(images=x1)['output'].numpy()[0]
   cos_y0_y1 = np.dot(y0, y1.T)/(norm(y0) * norm(y1))
   
   print(round(cos_y0_y1, 4))
