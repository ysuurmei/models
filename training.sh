# Set up the working environment.
CURRENT_DIR=$(pwd)
cd research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..
WORK_DIR="${CURRENT_DIR}/research/deeplab"
DATASET_DIR="/home/ubuntu/data_imat/dl_dataset"

# Set up the working directories.
NUM_ITERATIONS=20000
PQR_FOLDER="PQR"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
DATASET="${DATASET_DIR}/${PQR_FOLDER}/tfrecord"

mkdir -p "${DATASET_DIR}/${PQR_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"


python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"