import numpy as np
import pandas as pd
import os
import json

SEED = 1234
np.random.seed(SEED)

# Set the model version and data folder as environmental variables, so that we can pass them to the .sh script
DATA_FOLDER = r'C:\Users\YoupSuurmeijer\Downloads\train'
DATA_FILE = 'train.csv'

# Set the parameters for the new dataset
SUBSET = ["shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "pants", "shorts",
          "skirt", "coat", "dress", "jumpsuit", "cape"]

# Load the label descripions file and subset the dataset based on the label description indices
with open(os.path.join(DATA_FOLDER, 'label_descriptions.json')) as json_data:
    label_descriptions = json.load(json_data)


# Read the labels file in a dataframe
labels = pd.read_csv(os.path.join(DATA_FOLDER, DATA_FILE))

# If required subset the dataset
if SUBSET:
    subset_indices = [i['id'] for i in label_descriptions['categories'] if i['name'] in SUBSET]
    subset_rows = [int(i.split('_')[0]) in subset_indices for i in labels['ClassId']]
    labels = labels[subset_rows]
    print('Subsetting dataset, using {} images'.format(len(labels['ImageId'].unique())))

labels['SimpleClass'] = [int(i.split('_')[0]) for i in labels['ClassId']]

labels.groupby('SimpleClass').count()
round(labels.groupby('SimpleClass').count().max()/labels.groupby('SimpleClass').count(),1)