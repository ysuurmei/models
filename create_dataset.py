import pandas as pd
import os
from PIL import Image
import json
import numpy as np
import shutil
import progressbar

def create_deeplab_dataset(root_folder, label_file, image_folder=None, subset = None, train_val_split=(0.9, 0.1)):

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

        mask = mask.reshape((height, width), order='F')
        mask_image = Image.fromarray(mask)
        mask_image.save(os.path.join(subdir_class, image))

        draw = np.random.choice([0,1], 1, p=train_val_split)[0]

        if draw:
            val_set.append(image)
        else:
            train_set.append(image)

        shutil.copyfile(os.path.join(image_folder, image), os.path.join(subdir_images, image))

    with open(os.path.join(subdir_sets, 'train.txt'), 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(os.path.join(subdir_sets, 'val.txt'), 'w') as f:
        for item in val_set:
            f.write("%s\n" % item)

    with open(os.path.join(subdir_sets, 'trainval.txt'), 'w') as f:
        for item in train_set+val_set:
            f.write("%s\n" % item)

if __name__ == '__main__':

    np.random.seed(1234)
    DATA_FOLDER = r'~/data_imat' # r'C:/Users/YoupSuurmeijer/Downloads'
    DATA_FILE = 'train.csv' #'train/train.csv'
    IMAGE_FOLDER = '~/data_imat/train' #r'C:\Users\YoupSuurmeijer\Downloads\dl_dataset\SegmentationClass'
    TRAIN_VAL_SPLIT = [0.9, 0.1]

    subset = ['pants', 'shorts', 'dress', 'shirt, blouse', 'sweater']

    with open(os.path.join(DATA_FOLDER, 'label_descriptions.json')) as json_data:
        label_descriptions = json.load(json_data)

    subset_indices = [i['id'] for i in label_descriptions['categories'] if i['name'] in subset]

    create_deeplab_dataset(root_folder=DATA_FOLDER, label_file=DATA_FILE,
                           image_folder=IMAGE_FOLDER, subset=subset_indices)

