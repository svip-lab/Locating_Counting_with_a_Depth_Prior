import cv2
import torch
import numpy as np
import scipy.io as sio
import tqdm
import sys
import csv
import math
from lib.core.config import cfg
import torchvision.transforms as transforms
from lib.utils.logging import setup_logging

logger = setup_logging(__name__)

class RGBDDataset():
    def initialize(self, opt):
        self.opt = opt
        self.dir = opt.dataroot
        if opt.phase == 'train':
            self.mode = True
            self.train_file = self.dir + 'train.csv'
        else:
            self.mode = False
            self.train_file = self.dir + 'test.csv'
        self.class_list = self.dir + 'class.csv'

        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.debug = self.opt.debug
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_datas = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_datas.keys())
        self.img_ids = list(self.image_datas.keys())
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
        A_path = A_path.replace('/p300/Dataset/', '/group/crowd_counting/')
        B_path = A_path.replace('img', 'depth').replace('IMG', 'GT').replace('.png', '.mat')
        A = cv2.imread(A_path)       # (1080, 1920, 3)
        B = sio.loadmat(B_path)
        B = B['depth']
        B[B >= 20000] = 20000
        B = B / 20000
        B[B < 0] = -1
        mask = (B < 0)
        mask = mask.astype(float)
        h, w = mask.shape
        M = np.float32([[1, 0, 0], [0, 1, 5]])
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        A_resize = cv2.resize(A, (480, 270))
        B_resize = cv2.resize(B, (480, 270), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (480, 270), interpolation=cv2.INTER_NEAREST)
        # if random.random() > 0.5:
        #     A_resize = np.flip(A_resize, axis=1)
        #     B_resize = np.flip(B_resize, axis=1)
        #     mask = np.flip(mask, axis=1)
        A_resize = A_resize.transpose((2, 0, 1))  # (3, 270, 480)
        B_resize = B_resize[np.newaxis, :, :]     # (1, 270, 480)
        mask = mask[np.newaxis, :, :]
        A_resize = self.scale_torch(A_resize, 255.).float()
        B_resize = torch.from_numpy(B_resize.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        B_bins = self.depth_to_bins_uniform(B_resize)
        invalid_side = [0, 0, 0, 0]
        resize_ratio = 1

        data = {'A': A_resize, 'A_raw': A, 'A_paths': A_path,
                'B': B_resize,'B_raw': B, 'B_bins': B_bins, 'B_paths': B_path, 'masks': mask,
                'invalid_side': np.array(invalid_side), 'ratio': np.float32(1.0 / resize_ratio)}
        return data

    def online_aug_test(self, anno_index):
        A_path = self.image_names[anno_index]
        A_path = A_path.replace('p300/', 'p300/data/')
        B_path = A_path.replace('img', 'depth').replace('IMG', 'GT').replace('.png', '.mat')
        A = cv2.imread(A_path)
        B = sio.loadmat(B_path)
        B = B['depth']
        B[B >= 20000] = 20000
        B = B / 20000
        B[B < 0] = -1
        mask = (B <= 0)
        mask = mask.astype(float)

        h, w = mask.shape
        M = np.float32([[1, 0, 0], [0, 1, 5]])
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        size = (480, 270)
        A_resize = cv2.resize(A, size)
        B_resize = cv2.resize(B, size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        A_resize = A_resize.transpose((2, 0, 1))  # (3, 270, 480)
        B_resize = B_resize[np.newaxis, :, :]     # (1, 270, 480)
        mask = mask[np.newaxis, :, :]
        A_resize = self.scale_torch(A_resize, 255.).float()
        B_resize = torch.from_numpy(B_resize.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        B_bins = self.depth_to_bins_uniform(B_resize)
        invalid_side = [0, 0, 0, 0]
        resize_ratio = 1

        data = {'A': A_resize, 'A_raw': A, 'A_paths': A_path,
                'B': B_resize,'B_raw': B, 'B_bins': B_bins, 'B_paths': B_path, 'masks': mask,
                'invalid_side': np.array(invalid_side), 'ratio': np.float32(1.0 / resize_ratio)}
        return data

    def load_point(self, image_index):
        annotation_list = self.image_datas[self.image_names[image_index]]
        locs = np.zeros((0, 2))
        for idx, a in enumerate(annotation_list):
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            loc = np.zeros((1, 2))
            loc[0, 0] = (x1 + x2) / 2
            loc[0, 1] = (y1 + y2) / 2
            locs = np.append(locs, loc, axis=0)
        return locs

    def depth_to_bins_uniform(self, depth):  # uniform
        invalid_mask = depth < 0.
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX

        cfg.DATASET.DEPTH_BIN_INTERVAL = (cfg.DATASET.DEPTH_MAX - cfg.DATASET.DEPTH_MIN) / cfg.MODEL.DECODER_OUTPUT_C
        # print(cfg.DATASET.DEPTH_MAX, cfg.DATASET.DEPTH_MIN, cfg.DATASET.DEPTH_BIN_INTERVAL)  # 1.0 0.01 0.0066

        bins = ((depth - cfg.DATASET.DEPTH_MIN) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        return bins

    def depth_to_bins_log(self, depth):  # log
        """Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]"""
        invalid_mask = depth < 0.
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX
        # print(cfg.DATASET.DEPTH_MAX, cfg.DATASET.DEPTH_MIN, cfg.DATASET.DEPTH_MIN_LOG)  # 3.5 0.01 -2.0
        # log10(cfg.DATASET.DEPTH_MAX)=0.5440680443502756 -> (og10(cfg.DATASET.DEPTH_MAX)-cfg.DATASET.DEPTH_MIN_LOG)/150
        # print(cfg.DATASET.DEPTH_BIN_INTERVAL)  # 0.016960453629001837 - uniform in log scale
        bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        return bins

    def scale_torch(self, img, scale):
        """Scale the image and output it in torch.tensor.
        :param img: input image. [C, H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]"""
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
        return 'RGBD'

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)
            if img_file not in result:
                result[img_file] = []
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            # result[img_file] - img_file[info] - info{'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name}
            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        for idx, a in enumerate(annotation_list):  # parse annotations
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            annotation = np.zeros((1, 5))
            # annotation
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2
            #annotation[0, 4] = self.name_to_label(a['class'])
            annotation[0, 4] = 1
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _open_for_csv(self, path):
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def _parse(self, value, function, fmt):
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def load_classes(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1
            try:
                # 'head', '0'
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))  # 0

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            # result['head'] = 0
            result[class_name] = class_id
        return result
