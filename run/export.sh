cd research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..

python research/deeplab/export_model.py \
  --logtostderr \
  --vis_split="val" \
  --checkpoint_path='/home/ubuntu/data_imat/dl_dataset/PQR/exp/train_on_trainval_set/train/model.ckpt-4343'  \
  --export_path=/home/ubuntu/data_imat/dl_dataset/PQR/exp/train_on_trainval_set/train/frozen_inference_graph.pb \
  --dataset = "imat_fashion" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --vis_crop_size=512 \
  --vis_crop_size=512 \
  --decoder_output_stride=4 \
  --output_stride=16

