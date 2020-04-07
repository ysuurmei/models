from urllib import request
import os
import tarfile

DOWNLOAD_PATH = r'/home/ubuntu/data_imat/deeplab/pretrained'

if not os.path.exists(DOWNLOAD_PATH):
    print('Directory', DOWNLOAD_PATH, 'not found, creating directory....')
    os.makedirs(DOWNLOAD_PATH)

MODEL_NAME = 'resnet_v1_50_beta_imagenet'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    'resnet_v1_50_beta_imagenet':
        'resnet_v1_50_2018_05_04.tar.gz'
}
_TARBALL_NAME = 'deeplab_model.tar.gz'
_UNZIP_FOLDER = _MODEL_URLS[MODEL_NAME].split('_20')[0]

# Set download path here!
file_path = os.path.join(DOWNLOAD_PATH, _TARBALL_NAME)

print('downloading model, this might take a while...')
request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                   file_path)

print('download completed! Unpacking model in ', _UNZIP_FOLDER)
tar_file = tarfile.open(file_path)
tar_file.extractall(DOWNLOAD_PATH)
print('Extraction completed!')
