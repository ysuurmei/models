import tarfile
import os
from PIL import Image
from deeplab import DeepLabModel


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

if __name__ == '__main__':
    import glob
    from imgaug.augmentables.segmaps import SegmentationMapOnImage
    import numpy as np
    from progressbar import progressbar
    from utils import SEGMAP_COLORS

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import json
    from background_remover_mrcnn import BackgroundRemover

    DATA_FOLDER = r'C:\Users\YoupSuurmeijer\Downloads\train'
    SUBSET = ["shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "pants", "shorts",
             "skirt", "coat", "dress", "jumpsuit", "cape"]

    # Load the label descripions file and subset the dataset based on the label description indices
    with open(os.path.join(DATA_FOLDER, 'label_descriptions.json')) as json_data:
        label_descriptions = json.load(json_data)

    subset_indices = [i['id'] for i in label_descriptions['categories'] if i['name'] in SUBSET]

    legend_elements = []
    for idx, el in enumerate(SUBSET):
        col=SEGMAP_COLORS[subset_indices[idx]+1]
        col=tuple(ti/255 for ti in col)
        element = Patch(color=col, label=el)
        legend_elements.append(element)

    PATH_MODEL = r'C:\Users\YoupSuurmeijer\Documents\models\test\models\model_v6_100000.tar.gz'
    PATH_IMAGES = r'C:\Users\YoupSuurmeijer\Documents\models\test\test_images\old_set'
    PATH_OUTPUT = os.path.join(r'C:\Users\YoupSuurmeijer\Documents\models\test\test_output',
                  os.path.basename(PATH_MODEL).split('.')[0])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    model = DeepLabModel(PATH_MODEL, logits = True, input_size=512)
    os.chdir(PATH_IMAGES)

    bg_remover = BackgroundRemover(model=r'C:\Users\YoupSuurmeijer\Documents\VIPO-project\3. Production\models\deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz')


    for image in progressbar(glob.glob('*.jpg')):
        img = Image.open(image)
        # Remove background
        img_no_bg = bg_remover.make_background_transparant(np.array(img))
        img_no_bg = Image.fromarray(img_no_bg.astype('uint8'))
        # Create segmap
        image2, segmap, batch_segmap = model.run(img_no_bg)
        overlay = SegmentationMapOnImage(segmap, shape=segmap.shape).draw_on_image(np.array(image2), alpha=0.75, colors=SEGMAP_COLORS)
        imgplot = plt.imshow(overlay[0])
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        path_output = os.path.join(PATH_OUTPUT, 'output_'+os.path.basename(image))
        plt.savefig(path_output, bbox_inches='tight', dpi=1200)
        plt.close()

