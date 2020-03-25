import tarfile
import os.path
from deeplab import DeepLabModel
from PIL import Image
import numpy as np
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

make_tarfile('tests/1k.tar.gz', r'C:\Users\YoupSuurmeijer\Downloads\model_1k')



model = DeepLabModel(r'C:\Users\YoupSuurmeijer\Documents\models\tests\1k.tar.gz')

im_pil = Image.open(r'C:\Users\YoupSuurmeijer\Documents\models\tests\test_images\IMG_20191127_155353.jpg')

# Run the prediction function
segmentation = model.run(im_pil)
mask = np.float32(segmentation[1])