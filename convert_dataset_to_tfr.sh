cd research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..

WORK_DIR="/home/ubuntu/data_imat/"
PQR_ROOT="${WORK_DIR}/dl_dataset"
SEMANTIC_SEG_FOLDER="${PQR_ROOT}/SegmentationClass"
# Build TFRecords of the dataset.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"
IMAGE_FOLDER="${PQR_ROOT}/JPEGImages"
LIST_FOLDER="${PQR_ROOT}/ImageSets"
echo "Converting PQR dataset..."
python research/deeplab/datasets/build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"