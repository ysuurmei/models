# Set up the working environment.
cd ../research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..

# Set up folder structure
MODEL_VERSION="v1"
WORK_DIR="/home/ubuntu/data_imat/deeplab"
DATASET_DIR="${WORK_DIR}/${MODEL_VERSION}"
PQR_FOLDER="PQR"
INIT_FOLDER="${WORK_DIR}/pretrained"
TRAIN_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/train"
EVAL_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/eval"
DATASET="${DATASET_DIR}/tfrecord"

mkdir -p "${EVAL_LOGDIR}"

python research/deeplab/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--fine_tune_batch_norm=False \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--max_number_of_evaluations=1