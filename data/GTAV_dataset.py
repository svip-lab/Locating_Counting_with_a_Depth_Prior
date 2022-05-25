import cv2
import torch
import os.path
import numpy as np
import tqdm

from lib.core.config import cfg
import torchvision.transforms as transforms
from lib.utils.logging import setup_logging
logger = setup_logging(__name__)

class GTAVDataset():
    def initialize(self, opt):
        self.opt = opt
        self.dir = opt.dataroot
        self.root = self.dir + '/processed/'
        if opt.phase == 'train':
            self.mode = True
        else:
            self.mode = False

        nums = 0
        self.image_datas = {}
        if self.mode:
            flist = self.dir + '/' + 'train_flist.txt'
            with open(flist, 'r') as f:
                lines = f.readlines()
                for line in tqdm.tqdm(lines):
                    scene, ind = line.split()
                    scene_dir = self.root + scene
                    img_file = scene_dir + '/pngs/' + ind + '.png'
                    if img_file not in self.image_datas:
                        self.image_datas[img_file] = []
            self.image_names = list(self.image_datas.keys())
            self.img_ids = self.image_names
        else:
            flist = self.dir + '/meta_test_flist.txt'
            with open(flist) as f:
                lines = f.readlines()
                for line in tqdm.tqdm(lines):
                    nums += 1
                    if nums > 1000:
                        break
                    scene, ind = line.split()
                    scene_dir = self.root + scene
                    img_file = scene_dir + '/pngs/' + ind + '.png'
                    if img_file not in self.image_datas:
                        self.image_datas[img_file] = []
            self.image_names = list(self.image_datas.keys())
            self.img_ids = self.image_names
        print('init ', len(self), ' data')

    def __getitem__(self, anno_index):
        while True:
            if 'train' in self.opt.phase:
                data = self.online_aug_train(anno_index)
            else:
                data = self.online_aug_test(anno_index)
            if data is None:
                anno_index = np.random.choice(len(self))
                continue
            return data

    def online_aug_train(self, anno_index):
        A_path = self.image_names[anno_index]
        B_path = A_path.replace('pngs', 'depth').replace('png', 'raw')
        if not os.path.exists(B_path):
            return None
        A = cv2.imread(A_path)
        B = self.load_depth(B_path)
        mask = (B > 0.1334)
        mask = mask.astype(float)

        size = (480, 270)
        A_resize = cv2.resize(A, size)
        B_resize = cv2.resize(B, size)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        # flip
        # if random.random() > 0.5:
        #     A_resize = np.flip(A_resize, axis=1)
        #     # cv2.imwrite('/root/vis/img_flip.jpg', A_resize)
        #     B_resize = np.flip(B_resize, axis=1)
        A_resize = A_resize.transpose((2, 0, 1))
        B_resize = B_resize[np.newaxis, :, :]
        mask = mask[np.newaxis, :, :]
        A_resize = self.scale_torch(A_resize, 255.).float()
        B_resize = torch.from_numpy(B_resize.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        B_bins = self.depth_to_bins(B_resize)
        invalid_side = [0, 0, 0, 0]
        resize_ratio = 1

        data = {'A': A_resize, 'A_raw': A, 'A_paths': A_path,
                'B': B_resize, 'B_raw': B, 'B_bins': B_bins, 'B_paths': B_path, 'masks': mask,
                'invalid_side': np.array(invalid_side), 'ratio': np.float32(1.0 / resize_ratio)}
        return data

    def online_aug_test(self, anno_index):
        A_path = self.image_names[anno_index]
        B_path = A_path.replace('pngs', 'depth').replace('png', 'raw')
        if not os.path.exists(B_path):
            return None
        A = cv2.imread(A_path)
        B = self.load_depth(B_path)
        mask = (B > 0.1334)
        mask = mask.astype(float)

        size = (480, 270)
        A_resize = cv2.resize(A, size)
        B_resize = cv2.resize(B, size)
        mask = cv2.resize(mask, size)
        A_resize = A_resize.transpose((2, 0, 1))
        B_resize = B_resize[np.newaxis, :, :]
        mask = mask[np.newaxis, :, :]
        A_resize = self.scale_torch(A_resize, 255.).float()
        B_resize = torch.from_numpy(B_resize.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        B_bins = self.depth_to_bins(B_resize)
        invalid_side = [0, 0, 0, 0]
        resize_ratio = 1

        data = {'A': A_resize, 'A_raw': A, 'A_paths': A_path,
                'B': B_resize, 'B_raw': B_resize, 'B_bins': B_bins, 'B_paths': B_path, 'masks': mask,  # B_raw
                'invalid_side': np.array(invalid_side), 'ratio': np.float32(1.0 / resize_ratio)}
        return data

    def load_depth(self, path):
        depth = np.fromfile(path, dtype=np.float32)
        depth = depth.reshape(1080, 1920)

        depth = 2 ** depth - 1
        depth = 1 / (depth + 1e-9)

        depth *= 0.001
        depth[depth > 1] = 1
        return depth

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth < 0.
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX

        cfg.DATASET.DEPTH_BIN_INTERVAL = (cfg.DATASET.DEPTH_MAX - cfg.DATASET.DEPTH_MIN) / cfg.MODEL.DECODER_OUTPUT_C

        bins = ((depth - cfg.DATASET.DEPTH_MIN) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        # invalid_mask = depth < 0.
        # depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        # depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX

        # bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        # bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        # bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        # depth[invalid_mask] = -1.0
        return bins

    def scale_torch(self, img, scale):
        """
        Scale the image and output it in torch.tensor.
        :param img: input image. [C, H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W
        """
        img = img.astype(np.float32)
        img /= scale
        img = torch.from_numpy(img.copy())
        if img.size(0) == 3:
            img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
        else:
            img = transforms.Normalize((0,), (1,))(img)
        return img

    def __len__(self):
        return len(self.image_names)

    def name(self):
        return 'GTAV'

