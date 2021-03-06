cd ../research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..

# Set up folder structure
MODEL_VERSION="v8"
CHECKPOINT="100000"
WORK_DIR="/home/ubuntu/data_imat/deeplab"
DATASET_DIR="${WORK_DIR}/${MODEL_VERSION}"
PQR_FOLDER="PQR"
TRAIN_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/train"

#Configure export params
CHECKPOINT_PATH="${TRAIN_LOGDIR}/model.ckpt-${CHECKPOINT}"
EXPORT_PATH="${TRAIN_LOGDIR}/frozen_inference_graph_${CHECKPOINT}.pb"

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