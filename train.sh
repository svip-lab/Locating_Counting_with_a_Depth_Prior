CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_GTAV_metric.py \
		--dataset GTAV --dataroot /group/crowd_counting/GTAV-ours/ \
		--cfg_file lib/configs/resnext50_32x4d_GTAV --lr 0.05 --batchsize 8
