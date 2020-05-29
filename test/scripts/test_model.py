import tarfile
import os
from PIL import Image
from deeplab import DeepLabModel
from datetime import datetime
from operator import itemgetter
import pandas as pd

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

class SegmapFilter():
    def __init__(self, area_threshold):
        self.area_threshold = area_threshold

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def filter_simple(self, segmap):
        sigmoids = self.sigmoid(segmap)
        return sigmoids

class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def load_settings(file):
    with open(file) as f:
        settings = DotDict(json.load(f))
    return settings

def select_body_from_body_parts(body_parts):
    """
    Function to select the right body for classification from a list of bodyparts
    :param body_parts: list of bodyparts
    :return: the optimal body for classifcation
    """
    # Initialize list of scores, in the end the body with the highest score gets returned
    scores = []

    #For each body in the bodyparts input list determine the score
    for body in body_parts:
        score = 0
        try:
            # We assume the person we want to classify is in the middle of the image, so a score of 0-10 is given for
            # how near a body is to the center of the image (0: completely on the edge, 10: dead center)
            image_center = body['image'].shape[0] / 2
            body_center = body['body'][0] + body['body'][2] / 2
            center_bonus_score = 12 - 12 * abs(image_center - body_center) / image_center
            # Add a bonus for the height of the body (0-10)
            height_bonus_score = 10 - 10 * abs(body['image'].shape[1] - body['body'][3]) / body['image'].shape[1]

            score += center_bonus_score
            score += height_bonus_score

            # Add a bonus point for at least one face being present in the image
            if 'face' in body.keys():
                score += 5
            # Add a bonus point for at least one eye being present (usually 1-3 eyes are detected within a face)
            if body['eyes'] > 0:
                score += 1
            # Add another bonus if the number of eyes is exactly 2, this makes the function more flexible
            if body['eyes'] == 2:
                score += 5

        except KeyError:
            pass
        scores.append((body, score))

    # Determine body with the maximum score and return that
    body_with_max_score = max(scores, key=itemgetter(1))[0]
    return body_with_max_score

def crop_image(image, crop):
    """
    Function to crop an image based on a rect

    Parameters
    ----------
    image: image or array
    crop: rect (x,y,w,h) of the area to be cropped

    Returns
    -------
    cropped image
    """
    x, y, w, h = crop
    return image[y:y + h, x:x + w]

def convert_to_pil(image):
    """
    Function to convert PIL image to open-cv image format

    Parameters
    ----------
    image: PIL image

    Returns
    -------
    open-cv image
    """

    return Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))

def filter_segmap(segmap, sets, th=None):
    summary = pd.DataFrame.from_items(zip(['unique', 'counts'], np.unique(segmap, return_counts=True)))
    for _set in sets:
        subset = summary[summary.unique.isin(_set)]
        if subset.shape[0] > 0:
            set_max = subset.unique[subset.counts.idxmax()]
            for i in _set-{set_max}:
                segmap[segmap==i] = set_max
        if th:
            for row in summary.iterrows():
                if row[1]['counts']/summary['counts'].sum() < th:
                    segmap[segmap == row[1]['unique']] = 0
    return segmap


if __name__ == '__main__':
    import glob
    from imgaug.augmentables.segmaps import SegmentationMapOnImage
    import numpy as np
    from progressbar import progressbar
    from utils import SEGMAP_COLORS
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import json
    from background_remover_mrcnn import BackgroundRemover
    from body_extractor import BodyExtractor

    DATA_FOLDER = r'C:\Users\YoupSuurmeijer\Downloads\train'
    SUBSET = ["shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "pants", "shorts",
             "skirt", "coat", "dress", "jumpsuit"]

    # Load the label descripions file and subset the dataset based on the label description indices
    with open(os.path.join(DATA_FOLDER, 'label_descriptions.json')) as json_data:
        label_descriptions = json.load(json_data)

    subset_indices = [i['id'] for i in label_descriptions['categories'] if i['name'] in SUBSET]

    labels = [
        {
            "id": 1,
            "name": "shirt, blouse",
            "supercategory": "upperbody"
        },
        {
            "id": 2,
            "name": "top, t-shirt, sweatshirt",
            "supercategory": "upperbody"
        },
        {
            "id": 3,
            "name": "sweater",
            "supercategory": "upperbody"
        },
        {
            "id": 4,
            "name": "cardigan",
            "supercategory": "upperbody"
        },
        {
            "id": 5,
            "name": "jacket",
            "supercategory": "upperbody"
        },
        {
            "id": 6,
            "name": "pants",
            "supercategory": "lowerbody"
        },
        {
            "id": 7,
            "name": "shorts",
            "supercategory": "lowerbody"
        },
        {
            "id": 8,
            "name": "skirt",
            "supercategory": "lowerbody"
        },
        {
            "id": 9,
            "name": "coat",
            "supercategory": "upperbody"
        },
        {
            "id": 10,
            "name": "dress",
            "supercategory": "upperbody"
        },
        {
            "id": 11,
            "name": "jumpsuit",
            "supercategory": "wholebody"
        }
    ]
    legend_elements = []
    for i in labels:
        col=SEGMAP_COLORS[i['id']]
        col=tuple(ti/255 for ti in col)
        element = Patch(color=col, label=i['name'])
        legend_elements.append(element)

    PATH_MODEL = r'C:\Users\YoupSuurmeijer\Documents\models\test\models\model_v8_130000.tar.gz'
    PATH_IMAGES = r'C:\Users\YoupSuurmeijer\Documents\models\test\test_images\new_set'
    PATH_OUTPUT = os.path.join(r'C:\Users\YoupSuurmeijer\Documents\models\test\test_output',
                  os.path.basename(PATH_MODEL).split('.')[0])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    model = DeepLabModel(PATH_MODEL, logits=True, input_size=256)
    os.chdir(PATH_IMAGES)

    settings = load_settings(r'C:\Users\YoupSuurmeijer\Documents\VIPO-project\3. Production\settings.json')
    bg_remover = BackgroundRemover(model=r'C:\Users\YoupSuurmeijer\Documents\VIPO-project\3. Production\models\deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz')
    extractor = BodyExtractor(settings.body_extractor, r'C:\Users\YoupSuurmeijer\Documents\VIPO-project\3. Production\models')

    for image in progressbar(glob.glob('*.jpg')):
        # img = Image.open(image)
        img = cv2.imread(image)
        # Extract all the bodyparts
        bodyparts = extractor.extract_body_parts(img, image)

        # From the list of bodies and bodyparts use select the right body for classification
        selected_bodyparts = select_body_from_body_parts(bodyparts)  # Ensures only 1 body is classified in image
        img = crop_image(img, selected_bodyparts['body'])

        # Remove background
        # img = bg_remover.make_background_transparant(np.array(img))
        img = convert_to_pil(img) # Check image colors!!!!!


        # Create segmap
        start = datetime.now()
        image2, segmap, batch_segmap = model.run(img)
        segmap = filter_segmap(segmap, [{1,2,3,4}, {8, 10}, {4,5,9}, {7,8}], th=None)
        kernel = np.ones((3, 3), np.uint8)
        segmap = cv2.erode(segmap, kernel, iterations=4)
        segmap = cv2.dilate(segmap, kernel, iterations=4)
        print(datetime.now()-start)
        overlay = SegmentationMapOnImage(segmap, shape=segmap.shape).draw_on_image(np.array(image2), alpha=0.75, colors=SEGMAP_COLORS)
        imgplot = plt.imshow(overlay[0])
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        path_output = os.path.join(PATH_OUTPUT, 'output_'+os.path.basename(image))
        plt.savefig(path_output, bbox_inches='tight', dpi=1200)
        plt.close()

