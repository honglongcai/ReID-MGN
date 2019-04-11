import os
import re
import math
import random
import collections
from torchvision import transforms
from torch.utils.data import dataset,dataloader,sampler
from torchvision.datasets.folder import default_loader

from opt import opt

class Data:
    def __init__(self):
        train_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.trainset = Market1501(train_transform, 'train',opt.data_path)
        print('train_len:', len(self.trainset))
        self.testset = Market1501(test_transform, 'test',opt.data_path)
        print('test_len:', len(self.testset))
        self.queryset = Market1501(test_transform, 'query',opt.data_path)
        print('query_len:', len(self.queryset))
        self.testset1 = Market1501(test_transform, 'test1', opt.data_path)
        print('test_len:', len(self.testset1))
        self.queryset1 = Market1501(test_transform, 'query1', opt.data_path)
        print('query_len:', len(self.queryset1))
        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid, batch_image=opt.batchimage),
                                                  batch_size=opt.batchid*opt.batchimage, num_workers=8,pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8,pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,pin_memory=True)
        self.test_loader1 = dataloader.DataLoader(self.testset1, batch_size=opt.batchtest, num_workers=8, pin_memory=True)
        self.query_loader1 = dataloader.DataLoader(self.queryset1, batch_size=opt.batchtest, num_workers=8, pin_memory=True)


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        elif dtype == 'query':
            self.data_path += '/query'
        elif dtype == 'test1':
            self.data_path += '/bounding_box_test1'
        elif dtype == 'query1':
            self.data_path += '/query1'

        self.imgs = [path for path in list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        #if opt.data_path == 'msmt17_suning55':
        return int(file_path.split('/')[-1].split('_')[1])
        #else:
         #   return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]


class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id

        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            self._id2index[_id].append(idx)

    def __iter__(self):
        unique_ids = self.data_source.unique_ids
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)