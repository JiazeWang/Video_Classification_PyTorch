# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ../main.py \
ucf101 \
../data/ucf101/train_ucf.txt \
../data/ucf101/val_ucf.txt \
--arch resnet50_3d \
--dro 0.4 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--epochs 90 \
--pretrained \
--batch-size 16 \
--lr 0.01 \
--lr_steps 40 80 \
--workers 16 \
--image_tmpl {:06d}.jpg \
-ef 5 \
# --resume output/ucf101_resnet50_3d_3D_length16_stride4_dropout0.4/checkpoint_10epoch.pth \
#--pretrained \
#--pretrained_model models/kinetics400_pre_3d_pre_2d/kinetics400_pre_3d_pre_2d.pth \
