cd research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..

python deeplab/export_model.py
  --checkpoint_path=/home/ubuntu/data_imat/dl_dataset/PQR/exp/train_on_trainval_set/train/model.ckpt-10000  \
  --export_path=/home/ubuntu/data_imat/dl_dataset/PQR/exp/train_on_trainval_set/train/frozen_inference_graph.pb \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18