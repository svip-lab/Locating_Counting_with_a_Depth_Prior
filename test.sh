CUDA_VISIBLE_DEVICES=0 python ./tools/test_RGBD_metric.py \
		--dataset RGBD --dataroot /p300/data/Dataset/SIST_RGBD/RGBDmerge_540P/Part_A/ \
		--cfg_file lib/configs/resnext50_32x4d_GTAV \
		--load_ckpt /p300/checkpoint.pth
