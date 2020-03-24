import urllib
import os
import tarfile

MODEL_NAME = 'xception_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

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
}
_TARBALL_NAME = 'deeplab_model.tar.gz'
_UNZIP_FOLDER = _MODEL_URLS[MODEL_NAME].split('_20')[0]
download_path = os.path.join(r'/home/ubuntu/data_imat/dl_dataset/PQR/exp/train_on_trainval_set', _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                   download_path)

print('download completed! Unpacking model in ', _UNZIP_FOLDER)
tar_file = tarfile.open(os.path.join(r'/home/ubuntu/data_imat/dl_dataset/PQR/exp/train_on_trainval_set', _TARBALL_NAME))
tar_file.extractall(os.path.join(r'/home/ubuntu/data_imat/dl_dataset/PQR/exp/train_on_trainval_set', _UNZIP_FOLDER))
print('Extraction completed!')
