import tarfile
import os
from PIL import Image
from deeplab import DeepLabModel


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

if __name__ == '__main__':
    import glob
    from line_profiler import LineProfiler
    from imgaug.augmentables.segmaps import SegmentationMapOnImage
    import numpy as np
    from progressbar import progressbar

    PATH_MODEL = r'C:\Users\YoupSuurmeijer\Documents\models\test\models\model_v1_50000.tar.gz'
    PATH_IMAGES = r'C:\Users\YoupSuurmeijer\Documents\models\test\test_images'
    PATH_OUTPUT = os.path.join(r'C:\Users\YoupSuurmeijer\Documents\models\test\test_output',
                  os.path.basename(PATH_MODEL).split('.')[0])

    if not os.path.exists(PATH_OUTPUT):
        os.mkdir(PATH_OUTPUT)

    lp = LineProfiler()
    #implement line profiler!

    model = DeepLabModel(PATH_MODEL)
    os.chdir(PATH_IMAGES)


    for image in progressbar(glob.glob('*.jpg')):
        img = Image.open(image)
        # Create segmap
        segmap = model.run(img)[1]
        overlay = SegmentationMapOnImage(segmap, shape=img.size).draw_on_image(np.array(img), alpha=0.75, resize="segmentation_map")
        img_output = Image.fromarray(overlay[0])
        path_output = os.path.join(PATH_OUTPUT, 'output_'+os.path.basename(image))
        img_output.save(path_output)

