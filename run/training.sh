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
MODEL_VERSION='v8'
NUM_ITERATIONS=130000

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
  --num_clones=2 \
  --save_summaries_secs=120\
  --dataset="imat_fashion" \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=257,257 \
  --train_batch_size=16 \
  --base_learning_rate=0.0001 \
  --end_learning_rate=0.000005 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${WORK_DIR}/pretrained/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=False \
  --label_weights=1 \
  --label_weights=5 \
  --label_weights=4 \
  --label_weights=8 \
  --label_weights=8 \
  --label_weights=4 \
  --label_weights=4 \
  --label_weights=6 \
  --label_weights=4 \
  --label_weights=6 \
  --label_weights=4 \
  --label_weights=8 \


CHECKPOINT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${TRAIN_LOGDIR}/frozen_inference_graph_${NUM_ITERATIONS}.pb"

python research/deeplab/export_model.py \
  --logtostderr \
  --checkpoint_path="${CHECKPOINT_PATH}"  \
  --export_path="${EXPORT_PATH}" \
  --dataset="imat_fashion" \
  --model_variant="xception_65" \
  --crop_size=257 \
  --crop_size=257 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --inference_scales=1.0 \
  --num_classes=12

tar -czvf "${TRAIN_LOGDIR}/model_${MODEL_VERSION}_${CHECKPOINT}.tar.gz" "${CHECKPOINT_PATH}.data-00000-of-00001" \
"${CHECKPOINT_PATH}.meta" "${CHECKPOINT_PATH}.index" "${EXPORT_PATH}"

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 stop-instances --instance-ids $INSTANCE_ID