# NOTE! Run this shell script as sudo user else won't stop ec2 instance at the end 'sudo bash training.sh'
# Also run 'sudo source activate tensorflow_p36' instead of regular 'source activate ...'

# Set up the working environment.
cd ..
CURRENT_DIR=$(pwd)
cd research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..

# Set working directory and version and iteration configs
WORK_DIR='/home/ubuntu/data_imat/deeplab'
MODEL_VERSION='v1'
NUM_ITERATIONS=100000

# Set up folder structure
DATASET_DIR="${WORK_DIR}/${MODEL_VERSION}"
PQR_FOLDER="PQR"
INIT_FOLDER="${WORK_DIR}/pretrained"
TRAIN_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/train"
DATASET="${DATASET_DIR}/tfrecord"

# Create directories
mkdir -p "${DATASET_DIR}/${PQR_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"

# copy this file as training_settings to directory head (for reference)
cp run/training.sh "${DATASET_DIR}/training_settings.txt"

# Run the python training script
python research/deeplab/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513,513 \
  --train_batch_size=4 \
  --base_learning_rate=0.0001 \
  --end_learning_rate=0.000001 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=False \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --initialize_last_layer=false \
  #--label_weights="[3.0, 1.2, 12.6, 16.9, 2.4, 25.9, 1.5, 6.8, 3.7, 0, 1.0]"



