# Set up the working environment.
CURRENT_DIR=$(pwd)
cd research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..
WORK_DIR="${CURRENT_DIR}/research/deeplab"
DATASET_DIR="/home/ubuntu/data_imat"

# Set up the working directories.
NUM_ITERATIONS=10000
PQR_FOLDER="dl_dataset/PQR"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}"
TRAIN_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
DATASET="${DATASET_DIR}/tfrecord"

python "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size=1025,2049 \
--fine_tune_batch_norm=False \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--max_number_of_evaluations=1