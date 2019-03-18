from __future__ import division
import numpy as np
import nibabel as nib
import random
import glob
import os

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class DataProvider(object):
    """
    DataProvider is the data interface for tensorflow model training, validation and testing

    functions:
    normalize_image
    """
    def __init__(self, conf):
        logging.info('initialize dataprovider...')

        # data dir
        self.data_patch_dir = conf.data_patch_dir
        self.seg_patch_dir = conf.seg_patch_dir
        self.data_volume_dir = conf.data_volume_dir
        self.seg_volume_dir = conf.seg_volume_dir

        # train list
        logging.info('get train patch list...')
        self.train_case_id_list = np.sort(conf.train_case_id_list)
        self.train_patch_list = self.get_train_patch_list()
        logging.info('########### train list #############')
        logging.info('train_case_id_list: {}'.format(self.train_case_id_list))
        logging.info('train_case_id_list length: {}'.format(len(self.train_case_id_list)))
        logging.info('train_patch_list length: {}'.format(len(self.train_patch_list)))
        logging.info('####################################')

        # val list
        self.val_case_id_list = np.sort(conf.val_case_id_list)
        logging.info('########### val list ###############')
        logging.info('val_case_id_list: {}'.format(self.val_case_id_list))
        logging.info('val_case_id_list length: {}'.format(len(self.val_case_id_list)))
        logging.info('####################################')

    def get_train_patch_list(self):
        train_patch_list = glob.glob(os.path.join(self.data_patch_dir, '*/*.npy'))

        # case id filter
        if len(self.train_case_id_list) > 0:
            train_patch_list = [x for x in train_patch_list if self.get_case_id_from_patch_path(x) in self.train_case_id_list]

        # for patch in train_patch_list:
        #     seg_patch_data = np.load(self.get_patch_seg_from_data(patch, self.seg_patch_dir))
        #     if np.max(seg_patch_data) == 0:
        #         if np.random.uniform() > self.background_keep_prob:
        #             train_patch_list.remove(patch)

        # train_patch_list.sort(key=lambda x: (x.split('/')[-1][:-4].split('_')[0], int(x.split('/')[-1][:-4].split('_')[1])))

        train_patch_list.sort()
        return train_patch_list

    def get_val_volume_list(self):
        val_volume_list = self.get_volume_list(self.val_case_id_list, self.data_volume_dir)

        logging.info('########### val list ##############')
        logging.info('val_volume_list: {}'.format(val_volume_list))
        logging.info('val_volume_list length: {}'.format(len(val_volume_list)))
        logging.info('####################################')

        return val_volume_list

    def get_volume_list(self, case_id_list, volume_dir):
        volume_list = glob.glob(os.path.join(volume_dir, '*.nii'))

        # case id filter
        if len(case_id_list) > 0:
            volume_list = [x for x in volume_list if self.get_case_id_from_volume_path(x) in case_id_list]

        volume_list.sort()
        return volume_list

    @staticmethod
    def get_case_id_from_volume_path(volume_path):
        """
        :param slice_fullpath: the full path of a slice
        :return: the case id of the slice
        """
        return volume_path.split('/')[-1][:-4]

    @staticmethod
    def get_case_id_from_patch_path(patch_path):
        """
        :param slice_fullpath: the full path of a slice
        :return: the case id of the slice
        """
        return patch_path.split('/')[-1][:-4][:-6]

    def get_random_batch(self, batch_size):
        """
        :param batch_size: the batch size for training
        :return: the data for training in the specified batch size
        """
        if batch_size < 0:
            raise Exception('batch size should be a positive integer')
        batch = random.sample(self.train_patch_list, batch_size)
        data_batch = []
        seg_batch = []
        for patch in batch:
            data, seg = self.read_patch(patch, with_seg=True)
            data_batch.append(data)
            seg_batch.append(seg)
        return np.asarray(data_batch, dtype=np.float32), np.asarray(seg_batch, dtype=np.uint8)

    def get_patch_seg_from_data(self, data_fullpath, seg_patch_dir):
        seg_filename = data_fullpath.split('/')[-1].replace('volume-', 'segmentation-')
        patch_seg_fullpath = os.path.join(os.path.join(seg_patch_dir, self.get_case_id_from_patch_path(data_fullpath).replace('volume-', 'segmentation-')), seg_filename)
        return patch_seg_fullpath

    def get_volume_seg_from_data(self, data_fullpath, volume_seg_dir):
        seg_filename = data_fullpath.split('/')[-1].replace('volume-', 'segmentation-')
        volume_seg_fullpath = os.path.join(volume_seg_dir, seg_filename)
        return volume_seg_fullpath

    def read_patch(self, patch_path, with_seg=False):
        data = np.load(patch_path)
        if with_seg:
            seg = np.load(self.get_patch_seg_from_data(patch_path, self.seg_patch_dir))
            return data, seg
        else:
            return data

    def read_volume(self, volume_path, with_seg=False):
        """
        Read one nii volume with name format of case_id.nii
        :param volume_path: the full path of a volume
        :return: the volume data, affine matrix, case_id
        """
        case_id = self.get_case_id_from_volume_path(volume_path)
        nii_volume = nib.load(volume_path)
        data_volume = nii_volume.get_data()
        affine = nii_volume.get_affine()
        if with_seg:
            seg_volume = nib.load(self.get_volume_seg_from_data(volume_path, self.seg_volume_dir)).get_data()
            return case_id, data_volume, seg_volume, affine
        else:
            return case_id, data_volume, affine

    @staticmethod
    def normalize_image(tensor):
        """
        normalize a tensor between [0, 1]
        :param tensor: a tensor in an arbitrary shape
        """
        max_value = np.max(tensor)
        min_value = np.min(tensor)
        return (tensor - min_value) / (max_value - min_value)

