from research.deeplab.datasets.build_data import  ImageReader
import tensorflow as tf
import os
import glob
print(glob.glob("/home/adam/*.txt"))
image_reader = ImageReader('jpeg', channels=3)
label_reader = ImageReader('png', channels=1)

IMAGE_FOLDER = r"/home/ubuntu/data_imat/dl_dataset/JPEGImages"
SEGMENTATION_FOLDER = r'/home/ubuntu/data_imat/dl_dataset/SegmentationClass'

for file in glob.glob('*.jpeg'):
    image_filename = os.path.join(
        IMAGE_FOLDER, file)
    seg_filename = os.path.join(
        SEGMENTATION_FOLDER,
        os.path.splitext(file)[0])
    print('Files: ', image_filename, seg_filename)
    image_data = tf.gfile.GFile(image_filename, 'rb').read()
    height, width = image_reader.read_image_dims(image_data)

    seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
    seg_height, seg_width = label_reader.read_image_dims(seg_data)
    if height != seg_height or width != seg_width:
        print('Error!')
        print(image_filename, height, width)
        print(seg_filename, seg_height, seg_width)
    else:
        print('OK!')
