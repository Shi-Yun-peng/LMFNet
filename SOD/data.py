import os
from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, dataname,dataclass, img_ids,transform=None):

        self.dataname=dataname
        self.dataclass = dataclass
        self.transform = transform
        self.ids=img_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(f'./data/{self.dataname}/{self.dataclass}/input', self.ids[idx] + '.jpg')).convert('RGB')
        label = Image.open(os.path.join(f'./data/{self.dataname}/{self.dataclass}/label', self.ids[idx]  + '.png')).convert('L')

        if self.transform is not None:
            [image, label] = self.transform(image, label)

        return image, label


import numpy as np
from PIL import Image
import pickle
import os.path as osp

class LoadData(object):
    '''
    Class to laod the data
    '''
    def __init__(self, dataname,dataclass,  cached_data_file,img_ids):
        '''
        :param data_dir: directory where the dataset is kept
        :param cached_data_file: location where cached file has to be stored
        '''
        self.dataname=dataname
        self.dataclass=dataclass
        # self.dataset = dataset
        self.cached_data_file = cached_data_file
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.img_ids=img_ids

    def process(self):
        '''
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        no_files = 0
        for ids in self.img_ids:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
            print(ids)
            train_img_file = os.path.join(f'./data/{self.dataname}/{self.dataclass}/input', ids+'.jpg')
            train_label_file = os.path.join(f'./data/{self.dataname}/{self.dataclass}/label', ids+'.png')
            rgb_train_img = Image.open(train_img_file).convert('RGB')
            rgb_train_img = np.array(rgb_train_img, dtype=np.float32)

            self.mean[0] += np.mean(rgb_train_img[:, :, 0])
            self.mean[1] += np.mean(rgb_train_img[:, :, 1])
            self.mean[2] += np.mean(rgb_train_img[:, :, 2])

            self.std[0] += np.std(rgb_train_img[:, :, 0])
            self.std[1] += np.std(rgb_train_img[:, :, 1])
            self.std[2] += np.std(rgb_train_img[:, :, 2])
            no_files += 1
        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        data_dict = dict()
        data_dict['mean'] = self.mean
        data_dict['std'] = self.std
        pickle.dump(data_dict, open(self.cached_data_file, 'wb'))

        return data_dict
