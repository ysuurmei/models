import pandas as pd
import os
from PIL import Image
import json
import numpy as np
import shutil
import progressbar
from itertools import chain
import multiprocessing

class internWorker():
    def __init__(self, _id, labels , dirs, input_size, train_val_split):
        self._id = _id
        self.labels = labels
        self.dirs = dirs
        self.input_size = input_size
        self.train_val_split = train_val_split
        self.train_set, self.val_set = [], []
        print('Process ', self._id, 'is created')

    def run(self, q_train_set, q_val_set):
        print('Thread ', self._id, 'started')

        for image in progressbar.progressbar(self.labels['ImageId'].unique()):

            # Load actual image
            rgb_image = Image.open(os.path.join(self.dirs['image_folder'], image))
            width, height = rgb_image.size

            # Subset all image labels and initialize mask
            image_labels = self.labels[self.labels['ImageId'] == image]
            mask = np.full(height * width, 0, dtype=np.uint8)

            # Combine all image masks into a single segmmap
            for index, row in image_labels.iterrows():
                annotation = [int(x) for x in row['EncodedPixels'].split(' ')]

                for i, start_pixel in enumerate(annotation[::2]):
                    mask[start_pixel: start_pixel + annotation[2 * i + 1]] = int(
                        row['ClassId'].split('_')[0]) + 1  # use ClassId + 1 because background = 0

            # Create mask image
            mask = mask.reshape((height, width), order='F')
            mask_image = Image.fromarray(mask)

            # Resize mask + actual image
            resize_ratio = 1.0 * self.input_size / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            rgb_image_resized = rgb_image.convert('RGB').resize(target_size, Image.ANTIALIAS)
            mask_image_resized = mask_image.resize(target_size, Image.ANTIALIAS)

            # Save mask + actual image
            mask_image_resized.save(os.path.join(self.dirs['subdir_class'], os.path.splitext(image)[0] + ".png"))
            rgb_image_resized.save(os.path.join(self.dirs['subdir_images'], image))

            # Randomly add them to train and validation sets
            draw = np.random.choice([0, 1], 1, p=self.train_val_split)[0]

            if draw:
                self.val_set.put(os.path.splitext(image)[0])
            else:
                self.train_set.put(os.path.splitext(image)[0])
        print('Thread ', self._id, 'completed')
        if not q_train_set.full() or not q_val_set.full():
            q_train_set.put(self.train_set)
            q_val_set.put(self.val_set)
        return

def create_deeplab_dataset_mp(model_version, root_folder, label_file, n_workers=8, image_folder=None, subset=None,
                           train_val_split=(0.9, 0.1),input_size=512, version_info=None):

    # initialize the required subdirectories and create them if they do not already exists
    subdir_sets = os.path.join(root_folder, model_version, 'ImageSets')
    subdir_images = os.path.join(root_folder, model_version, 'JPEGImages')
    subdir_class = os.path.join(root_folder, model_version, 'SegmentationClass')

    for directory in [os.path.join(root_folder, model_version), subdir_sets, subdir_images, subdir_class]:
        if not os.path.exists(directory):
            print('Directory', directory, 'not found, creating directory....')
            os.makedirs(directory)
        else:
            print('Directory', directory, 'already exists, clearing out existing files')
            try:
                shutil.rmtree(directory)
            except:
                pass

    # If version info is provided write info to text file in folder root directory
    if version_info:
        with open(os.path.join(root_folder, model_version, 'version_info.txt'), 'w') as f:
            f.write(version_info)

    # Read the labels file in a dataframe
    labels = pd.read_csv(os.path.join(root_folder, label_file))

    # If required subset the dataset
    if subset:
        subset_rows = [int(i.split('_')[0]) in subset for i in labels['ClassId']]
        labels = labels[subset_rows]
        print('Subsetting dataset, using {} images'.format(len(labels['ImageId'].unique())))

    dirs = {'image_folder': os.path.join(root_folder, image_folder),
            'subdir_class': subdir_class,
            'subdir_images': subdir_images}

    splits = np.array_split(labels, n_workers)
    q_train_set = multiprocessing.Queue()
    q_val_set = multiprocessing.Queue()
    procs = []

    for idx, split in enumerate(splits):
        worker = internWorker(idx, split, dirs, input_size, train_val_split) #(self, id, labels , dirs, input_size, train_val_split)
        proc = multiprocessing.Process(target=worker.run, args=(q_train_set, q_val_set,))
        procs.append(proc)
        proc.start()
    print('processing completed')
    # complete the processes
    for proc in procs:
        proc.join()

    print('\npopping items from queue:')
    train_set, val_set = [], []

    while not q_train_set.empty():
        train_set.append(q_train_set.get())

    while not q_val_set.empty():
        val_set.append(q_val_set.get())

    train_set = list(chain.from_iterable(train_set))
    val_set = list(chain.from_iterable(val_set))

    # Write train, val and trainval sets to .txt files
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
    from datetime import datetime

    # Set the seed to pin the random selection of train/val split
    SEED = 1234
    np.random.seed(SEED)

    # Set the model version and data folder as environmental variables, so that we can pass them to the .sh script
    os.environ['DATA_FOLDER'] = '/home/ubuntu/data_imat'
    os.environ['MODEL_VERSION'] = 'deeplab/v4'

    if not os.path.exists(os.path.join(os.environ['DATA_FOLDER'], os.environ['MODEL_VERSION'])):
        os.makedirs(os.path.join(os.environ['DATA_FOLDER'], os.environ['MODEL_VERSION']))

    # Set the location of the image files and labels
    DATA_FILE = 'train.csv'
    IMAGE_FOLDER = os.path.join(os.environ['DATA_FOLDER'], 'train')

    # Set the parameters for the new dataset
    # SUBSET = ["shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "pants", "shorts",
    #          "skirt", "coat", "dress", "jumpsuit", "cape", "glasses", "hat", "watch", "shoe", "bag, wallet"]
    SUBSET = ["jumpsuit"]
    TRAIN_VAL_SPLIT = [0.9, 0.1]

    # Load the label descripions file and subset the dataset based on the label description indices
    with open(os.path.join(os.environ['DATA_FOLDER'], 'label_descriptions.json')) as json_data:
        label_descriptions = json.load(json_data)

    subset_indices = [i['id'] for i in label_descriptions['categories'] if i['name'] in SUBSET]

    version_info = 'Version: {} \n Ceated on: {} \n Subset: {} \n train/val split: {} \n seed: {}'.format(
        os.environ['MODEL_VERSION'],
        datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        str(SUBSET),
        str(TRAIN_VAL_SPLIT),
        SEED
    )
    start = datetime.now()
    # Create the dataset in deeplab format
    print("Number of cpu : ", multiprocessing.cpu_count())
    create_deeplab_dataset_mp(model_version=os.environ['MODEL_VERSION'], root_folder=os.environ['DATA_FOLDER'],
                           label_file=DATA_FILE, image_folder=IMAGE_FOLDER, input_size=256,
                           subset=subset_indices, version_info=version_info, n_workers=multiprocessing.cpu_count())


    print(datetime.now()-start)
    # Run the shell script to convert the deeplab dataset to TFrecord format
    os.system('./convert_dataset_to_tfr.sh $DATA_FOLDER $MODEL_VERSION')
    # NOTE in case of permission issues run following command to change file permissions
    # 'chmod a+x convert_dataset_to_tfr.sh'

