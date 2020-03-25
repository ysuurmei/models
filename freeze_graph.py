from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph('~/data_imat/dl_dataset/PQR/exp/train_on_trainval_set/train/graph.pbtxt', "", False,
                          '~/data_imat/dl_dataset/PQR/exp/train_on_trainval_set/train/model.ckpt-10000', "output/softmax",
                           "save/restore_all", "save/Const:0",
                           '~/data_imat/dl_dataset/PQR/exp/train_on_trainval_set/train/frozentensorflowModel.pb', True, ""
                         )