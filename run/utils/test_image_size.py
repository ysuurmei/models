from research.deeplab.datasets.build_data import  ImageReader
import tensorflow as tf
import os
import glob
import progressbar



DATA_FOLDER = '/home/ubuntu/data_imat'
MODEL_VERSION = 'deeplab/v1'

IMAGE_FOLDER = os.path.join(DATA_FOLDER, MODEL_VERSION, "JPEGImages")
SEGMENTATION_FOLDER = os.path.join(DATA_FOLDER, MODEL_VERSION, "SegmentationClass")

image_reader = ImageReader('jpeg', channels=3)
label_reader = ImageReader('png', channels=1)
os.chdir(IMAGE_FOLDER)
print('Current directory: ', os.curdir)

for file in  progressbar.progressbar(glob.glob('*.jpg')):
    image_filename = os.path.join(
        IMAGE_FOLDER, file)
    seg_filename = os.path.join(
        SEGMENTATION_FOLDER,
        os.path.splitext(file)[0]+'.png')
    image_data = tf.gfile.GFile(image_filename, 'rb').read()
    height, width = image_reader.read_image_dims(image_data)

    seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
    seg_height, seg_width = label_reader.read_image_dims(seg_data)
    if height != seg_height or width != seg_width:
        print('Files: ', image_filename, seg_filename)
        print('Error!')
        print(image_filename, height, width)
        print(seg_filename, seg_height, seg_width)

