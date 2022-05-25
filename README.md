# Locating_Counting_with_a_Depth_Prior
[TPAMI] Locating and Counting Heads in Crowds With a Depth Prior [[Paper]](https://ieeexplore.ieee.org/document/9601215/)

## Dataset
Download or generate the virtual dataset from the [ShanghaiTechRGBDSyn](https://github.com/svip-lab/ShanghaiTechRGBDSyn) repository.

Download ShanghaiTechRGBD dataset from [OneDrive](https://yien01-my.sharepoint.com/:f:/g/personal/doubility_z0_tn/EhY4Svr1rRlDi7apZTtpepQBJejNSSYnQk1UNSqxhQ3jqA?e=RdhCtz).

## Depth Completion

## Style Transfer
[SG-GAN](https://github.com/Peilun-Li/SG-GAN).

## Train

## Inference
```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/test_RGBD_metric.py \
		--dataset RGBD --dataroot /p300/data/Dataset/SIST_RGBD/RGBDmerge_540P/Part_A/ \
		--cfg_file lib/configs/resnext50_32x4d_GTAV \
		--load_ckpt /p300/checkpoint.pth
```

ResNext50 backbone [[checkpoint_270x480]](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/chenxn1_shanghaitech_edu_cn/EWCAditWMiRDk8yOnHWBmZEBZRy1c_noaTHGEnlxWVnrKQ?e=MvCO5G)

## Acknowledgements

This repository borrows partially from [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction) and [MAML](https://github.com/katerakelly/pytorch-maml).

## Citation

If you find this repository useful for your research, please use the following:

```
@article{lian2021locating,
  title={Locating and Counting Heads in Crowds With a Depth Prior},
  author={Lian, Dongze and Chen, Xianing and Li, Jing and Luo, Weixin and Gao, Shenghua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```
