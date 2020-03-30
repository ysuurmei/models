cd ../research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..

# Set up folder structure
WORK_DIR="${CURRENT_DIR}/research/deeplab"
DATASET_DIR="${WORK_DIR}/${MODEL_VERSION}"
PQR_FOLDER="PQR"
INIT_FOLDER="${WORK_DIR}/pretrained"
TRAIN_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/train"

#Configure export params
CHECKPOINT=4343
CHECKPOINT_PATH="${TRAIN_LOGDIR}/model.ckpt-${CHECKPOINT}"
EXPORT_PATH="${TRAIN_LOGDIR}/frozen_inference_graph_${CHECKPOINT}.pb"

python research/deeplab/export_model.py \
  --logtostderr \
  --vis_split="val" \
  --checkpoint_path=CHECKPOINT_PATH  \
  --export_path=EXPORT_PATH \
  --dataset="imat_fashion" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --vis_crop_size=512 \
  --vis_crop_size=512 \
  --decoder_output_stride=4 \
  --output_stride=16

