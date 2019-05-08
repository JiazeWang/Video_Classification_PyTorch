# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ../main.py \
something \
../data/something/something_train.txt \
../data/something/something_val.txt \
--arch resnet50_3d \
--dro 0.4 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--epochs 110 \
--batch-size 16 \
--lr 0.01 \
--lr_steps 60 90 100 \
--workers 16 \
--image_tmpl {:05d}.jpg \
-ef 5 \
# --resume output/ucf101_resnet50_3d_3D_length16_stride4_dropout0.4/checkpoint_10epoch.pth \
#--pretrained \
#--pretrained_model models/kinetics400_pre_3d_pre_2d/kinetics400_pre_3d_pre_2d.pth \
