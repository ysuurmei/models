import tarfile
import os.path
from deeplab import DeepLabModel
from PIL import Image
import numpy as np
from datetime import datetime
import glob

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

make_tarfile('tests/4k.tar.gz', r'C:\Users\YoupSuurmeijer\Downloads\model_4k')

model = DeepLabModel(r'C:\Users\YoupSuurmeijer\Documents\models\tests\4k.tar.gz')

im_pil = Image.open(r'C:\Users\YoupSuurmeijer\Documents\models\tests\test_images\IMG_20191127_155353.jpg')

os.chdir(r'C:\Users\YoupSuurmeijer\Documents\models\tests\test_images')

for file in glob.glob('*.jpg'):
    im_pil = Image.open(file)
    start = datetime.now()
    # Run the prediction function
    segmentation = model.run(im_pil)
    print('Classification took: ')
    print(datetime.now()-start)
    if segmentation[1].max() > 0:
        print('YEEEY!')
        break