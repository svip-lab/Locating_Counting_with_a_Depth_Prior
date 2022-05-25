# import sys
# sys.path.append('/root/VNL_Meta/')
import torch
import torchvision
import numpy as np
import tqdm
import os
import cv2

from lib.utils.net_tools import load_ckpt
from tools.parse_arg_test import TestOptions
from lib.core.config import merge_cfg_from_file
from data.load_dataset import CustomerDataLoader
from lib.models.image_transfer import resize_image
from lib.utils.evaluate_depth_error import evaluate_err
from lib.models.metric_depth_model import MetricDepthModel
from lib.utils.logging import setup_logging, SmoothedValue

logger = setup_logging(__name__)

if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

    data_loader = CustomerDataLoader(test_args)
    test_datasize = len(data_loader)
    logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))

    model = MetricDepthModel()
    model.eval()

    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    # smoothed_absRel = SmoothedValue(test_datasize)
    # smoothed_rms = SmoothedValue(test_datasize)
    # smoothed_logRms = SmoothedValue(test_datasize)
    # smoothed_squaRel = SmoothedValue(test_datasize)
    # smoothed_silog = SmoothedValue(test_datasize)
    # smoothed_silog2 = SmoothedValue(test_datasize)
    # smoothed_log10 = SmoothedValue(test_datasize)
    # smoothed_delta1 = SmoothedValue(test_datasize)
    # smoothed_delta2 = SmoothedValue(test_datasize)
    # smoothed_delta3 = SmoothedValue(test_datasize)
    # smoothed_whdr = SmoothedValue(test_datasize)
    # smoothed_criteria = {'err_absRel': smoothed_absRel, 'err_squaRel': smoothed_squaRel, 'err_rms': smoothed_rms,
    #                      'err_silog': smoothed_silog, 'err_logRms': smoothed_logRms, 'err_silog2': smoothed_silog2,
    #                      'err_delta1': smoothed_delta1, 'err_delta2': smoothed_delta2, 'err_delta3': smoothed_delta3,
    #                      'err_log10': smoothed_log10, 'err_whdr': smoothed_whdr}

    for i, data in tqdm.tqdm(enumerate(data_loader)):
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['b_fake'])  # [540, 960]

        save_folder = '/p300/VNL_RGBD/RGBDfinetune/vis/'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            print('save to:', save_folder)
        img_path = data['A_paths'][0]
        ind1 = img_path.rfind('_')
        ind2 = img_path.rfind('.png')
        index = img_path[ind1 + 1: ind2]

        input = data['A'][0]
        inv_normalize = torchvision.transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        input = inv_normalize(input)
        input = input.permute(1, 2, 0).numpy()
        cv2.imwrite(save_folder + index + 'rgb.jpg', input * 255)  # RGB
        raw = data['B_raw'][0].numpy()
        cv2.imwrite(save_folder + index + 'raw.jpg', raw * 255)  # raw
        pred = pred_depth.cpu().numpy()
        cv2.imwrite(save_folder + index + 'pred.jpg', pred * 255)  # pred

    # print("###############absREL ERROR: %f", smoothed_criteria['err_absRel'].GetGlobalAverageValue())
    # print("###############silog ERROR: %f", np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (
    #     smoothed_criteria['err_silog'].GetGlobalAverageValue()) ** 2))
    # print("###############log10 ERROR: %f", smoothed_criteria['err_log10'].GetGlobalAverageValue())
    # print("###############RMS ERROR: %f", np.sqrt(smoothed_criteria['err_rms'].GetGlobalAverageValue()))
    # print("###############delta_1 ERROR: %f", smoothed_criteria['err_delta1'].GetGlobalAverageValue())
    # print("###############delta_2 ERROR: %f", smoothed_criteria['err_delta2'].GetGlobalAverageValue())
    # print("###############delta_3 ERROR: %f", smoothed_criteria['err_delta3'].GetGlobalAverageValue())
    # print("###############squaRel ERROR: %f", smoothed_criteria['err_squaRel'].GetGlobalAverageValue())
    # print("###############logRms ERROR: %f", np.sqrt(smoothed_criteria['err_logRms'].GetGlobalAverageValue()))
