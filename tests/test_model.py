import tarfile
import os.path
from deeplab import DeepLabModel
from PIL import Image

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

#make_tarfile('10k.tar.gz', r'C:\Users\YoupSuurmeijer\Downloads\model_10k')



model = DeepLabModel(r'C:\Users\YoupSuurmeijer\Documents\models\tests\10k.tar.gz')

im_pil = Image.open('test_images/IMG_20191121_143653.jpg')

# Run the prediction function
segmentation = model.run(im_pil)