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
MODEL_VERSION='v5'
NUM_ITERATIONS=10000

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
  --num_clones=1 \
  --dataset="imat_fashion" \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=257,257 \
  --train_batch_size=4 \
  --base_learning_rate=0.00001 \
  --end_learning_rate=0.000001 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=False \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --initialize_last_layer=false \




