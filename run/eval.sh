# Set up the working environment.
cd ../research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..

# Set up folder structure
MODEL_VERSION="v5"
WORK_DIR="/home/ubuntu/data_imat/deeplab"
DATASET_DIR="${WORK_DIR}/${MODEL_VERSION}"
PQR_FOLDER="PQR"
TRAIN_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/train"
EVAL_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/eval"
DATASET="${DATASET_DIR}/tfrecord"

mkdir -p "${EVAL_LOGDIR}"

python research/deeplab/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="mobilenet_v2" \
--output_stride=16 \
--eval_crop_size=257,257 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--max_number_of_evaluations=1 \
--dataset="imat_fashion" \