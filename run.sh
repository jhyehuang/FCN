#生成数据集
python ./code/convert_fcn_dataset.py --data_dir=/d/GitHub/data/W10_data/VOCdevkit/VOC2012 --output_dir=/d/GitHub/data/W10_data/train_data
#训练和验证
python drive/Colaboratory/code/train.py --checkpoint_path drive/Colaboratory/w10_data/vgg_16.ckpt --output_dir drive/Colaboratory/output --dataset_train drive/Colaboratory/w10_data/fcn_train.record --dataset_val drive/Colaboratory/w10_data/fcn_val.record --batch_size 16 --max_steps 2000