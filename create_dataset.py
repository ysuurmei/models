import pandas as pd
import os
from PIL import Image
import json
import numpy as np
import shutil
import progressbar

def create_deeplab_dataset(root_folder, label_file, image_folder=None, subset = None, train_val_split=(0.9, 0.1),input_size=512):

    train_set, val_set = [], []

    subdir_sets = os.path.join(root_folder, 'dl_dataset', 'ImageSets')
    subdir_images = os.path.join(root_folder, 'dl_dataset', 'JPEGImages')
    subdir_class = os.path.join(root_folder, 'dl_dataset', 'SegmentationClass')

    for directory in [os.path.join(root_folder, 'dl_dataset'), subdir_sets, subdir_images, subdir_class]:
        if not os.path.exists(directory):
            print('Directory', directory, 'not found, creating directory....')
            os.mkdir(directory)

    labels = pd.read_csv(os.path.join(root_folder, label_file))

    if subset:
        subset_rows = [int(i.split('_')[0]) in subset for i in labels['ClassId']]
        labels = labels[subset_rows]
        print('Subsetting dataset, using {} images'.format(len(labels['ImageId'].unique())))


    for image in  progressbar.progressbar(labels['ImageId'].unique()):
        image_labels = labels[labels['ImageId']==image]
        width, height = image_labels['Width'].iloc[0], image_labels['Height'].iloc[0]
        mask = np.full(height * width, 0, dtype=np.uint8)

        for index, row in image_labels.iterrows():
            annotation = [int(x) for x in row['EncodedPixels'].split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                mask[start_pixel: start_pixel + annotation[2 * i + 1]] = row['ClassId'].split('_')[0]

        #Create mask image
        mask = mask.reshape((height, width), order='F')
        mask_image = Image.fromarray(mask)

        #Load actual image
        rgb_image = Image.open(os.path.join(image_folder, image))
        #Resize mask + actual image
        width, height = image.size
        resize_ratio = 1.0 * input_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        rgb_image_resized = rgb_image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        mask_image_resized = mask_image.resize(target_size, Image.ANTIALIAS)

        # Save mask + actual image
        mask_image_resized.save(os.path.join(subdir_class, os.path.splitext(image)[0]+".png"))
        rgb_image_resized.save(os.path.join(subdir_images, image))

        #Randomly add them to train and validation sets
        draw = np.random.choice([0,1], 1, p=train_val_split)[0]

        if draw:
            val_set.append(os.path.splitext(image)[0])
        else:
            train_set.append(os.path.splitext(image)[0])

    with open(os.path.join(subdir_sets, 'train.txt'), 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    # Write train, val and trainval sets to .txt files
    with open(os.path.join(subdir_sets, 'val.txt'), 'w') as f:
        for item in val_set:
            f.write("%s\n" % item)

    with open(os.path.join(subdir_sets, 'trainval.txt'), 'w') as f:
        for item in train_set+val_set:
            f.write("%s\n" % item)

if __name__ == '__main__':

    np.random.seed(1234)
    DATA_FOLDER = r'/home/ubuntu/data_imat' # r'C:/Users/YoupSuurmeijer/Downloads'
    DATA_FILE = 'train.csv' #'train/train.csv'
    IMAGE_FOLDER = '/home/ubuntu/data_imat/train' #r'C:\Users\YoupSuurmeijer\Downloads\dl_dataset\SegmentationClass'
    TRAIN_VAL_SPLIT = [0.9, 0.1]

    subset = ['pants', 'dress', 'sweater']

    with open(os.path.join(DATA_FOLDER, 'label_descriptions.json')) as json_data:
        label_descriptions = json.load(json_data)

    subset_indices = [i['id'] for i in label_descriptions['categories'] if i['name'] in subset]

    create_deeplab_dataset(root_folder=DATA_FOLDER, label_file=DATA_FILE,
                           image_folder=IMAGE_FOLDER, subset=subset_indices)

