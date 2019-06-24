# -*- coding: utf-8 -*-

import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation

from config import config as cfg
from dataset import DatasetRegistry, DatasetSplit


def _category_id_to_class_id(category_id):
    if category_id in [1, 2, 3, 4, 5, 6]:
        return 1
    elif category_id in [7, 8, 9]:
        return 2
    elif category_id in [10, 11, 12, 13]:
        return 3

def _get_bb(img_path, anno_path, heavy_check=False):
    raw_json = json.loads(open(anno_path, 'r').read())
    if heavy_check:
        raw_img = cv2.imread(img_path)
        height, width, _ = raw_img.shape
    boxes = []
    classes = []
    for obj in [raw_json[key] for key in raw_json.keys() if 'item' in key]:
        bb = obj['bounding_box']
        # Note that DeepFashion2 format is bottom left and top right lul
        if heavy_check:
            x_top = np.clip(float(bb[0]), 0, width)
            y_top = np.clip(float(bb[3]), 0, height)
            x_bot = np.clip(float(bb[2]), 0, width)
            y_bot = np.clip(float(bb[1]), 0, height)
        else:
            x_top = float(bb[0])
            y_top = float(bb[3])
            x_bot = float(bb[2])
            y_bot = float(bb[1])
        boxes.append([x_top, y_top, x_bot, y_bot])
        classes.append(_category_id_to_class_id(obj['category_id']))
    return np.array(boxes, dtype=np.float32), np.array(classes)


class DeepFashion2Detection(DatasetSplit):
    """
    A class to load datasets, evaluate results for a datast split (e.g., "coco_train_2017")

    To use your own dataset that's not in COCO format, write a subclass that
    implements the interfaces.
    """
    class_names = ['top', 'bottom', 'long']
    cfg.DATA.CLASS_NAMES = ["BG"] + class_names

    def __init__(self, root_path, split):
        self.split = split
        self.data_path = os.path.join(root_path, split)

    def _load(self, load_anno=True, heavy_check=False):
        all_imgs = glob.glob(os.path.join(self.data_path, 'image', '*.jpg'))
        images_list = []
        annos_list = []
        for each_img in all_imgs:
            if heavy_check:
                img = cv2.imread(each_img)
                condition = img is not None
            else:
                condition = os.path.isfile(each_img)
            if not condition:
                print('[WARNING]:', each_img, 'not existed or empty. Skipped')
                continue
            file_name = each_img.split('/')[-1]
            each_anno = os.path.join(self.data_path, 'annos', file_name.replace('.jpg', '.json'))
            if not os.path.isfile(each_anno):
                print("[WARNING]: Can't find annotation for", file_name, "at", each_anno, ". Skipped")
                continue
            images_list.append(each_img)
            annos_list.append(each_anno)
        print('Found', len(images_list), 'images for', self.split)
        roidbs = []
        for img, anno in tqdm(zip(images_list, annos_list), desc=self.split, total=len(images_list)):
            img_id = '{}_{}'.format(img.split('/')[-1].replace('.jpg',''), type)
            sample = {
                'file_name': img,
                'img_id': img_id,
            }
            if load_anno:
                boxes, classes = _get_bb(img, anno)
                sample['boxes'] = boxes
                sample['class'] = classes
                sample['is_crowd'] = False
            roidbs.append(sample)
        return roidbs


    def training_roidbs(self):
        """
        Returns:
            roidbs (list[dict]):

        Produce "roidbs" as a list of dict, each dict corresponds to one image with k>=0 instances.
        and the following keys are expected for training:

        file_name: str, full path to the image
        boxes: numpy array of kx4 floats, each row is [x1, y1, x2, y2]
        class: numpy array of k integers, in the range of [1, #categories], NOT [0, #categories)
        is_crowd: k booleans. Use k False if you don't know what it means.
        segmentation: k lists of numpy arrays (one for each instance).
            Each list of numpy arrays corresponds to the mask for one instance.
            Each numpy array in the list is a polygon of shape Nx2,
            because one mask can be represented by N polygons.

            If your segmentation annotations are originally masks rather than polygons,
            either convert it, or the augmentation will need to be changed or skipped accordingly.

            Include this field only if training Mask R-CNN.
        """
        return self._load()


    def inference_roidbs(self):
        """
        Returns:
            roidbs (list[dict]):

            Each dict corresponds to one image to run inference on. The
            following keys in the dict are expected:

            file_name (str): full path to the image
            image_id (str): an id for the image. The inference results will be stored with this id.
        """
        return self._load(load_anno=False)


    def eval_inference_results(self, results, output=None):
        """
        Args:
            results (list[dict]): the inference results as dicts.
                Each dict corresponds to one __instance__. It contains the following keys:

                image_id (str): the id that matches `inference_roidbs`.
                category_id (int): the category prediction, in range [1, #category]
                bbox (list[float]): x1, y1, x2, y2
                score (float):
                segmentation: the segmentation mask in COCO's rle format.
            output (str): the output file or directory to optionally save the results to.

        Returns:
            dict: the evaluation results.
        """
        raise NotImplementedError


def register_deep_fashion_2(basedir):
    """
    Add COCO datasets like "coco_train201x" to the registry,
    so you can refer to them with names in `cfg.DATA.TRAIN/VAL`.
    """
    for split in ['train', 'val']:
        DatasetRegistry.register(split, lambda x=split: DeepFashion2Detection(basedir, x))


if __name__ == '__main__':
    basedir = '/Users/linus/techainer/DeepFashion/DeepFashion2'
    c = DeepFashion2Detection(basedir, 'train')
    roidb = c.training_roidbs()
    print("#Images:", len(roidb))
