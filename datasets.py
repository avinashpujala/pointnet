import torch
from torch.utils.data import Dataset
import pandas as pd
from random import random


class PointDataset(Dataset):
    def __init__(self, data_path, train=True, test=False, noise=False):
        super(PointDataset, self).__init__()
        self.data_path = data_path
        self.metadata = pd.read_csv(data_path + 'metadata_modelnet10.csv')
        if train:
            self.metadata = self.metadata.loc[self.metadata.split == 'train']
            self.metadata.reset_index(inplace=True)
        if test:
            self.metadata = self.metadata.loc[self.metadata.split == 'test']
            self.metadata.reset_index(inplace=True)
        self.metadata = self.metadata.loc[self.metadata['class'] != '.DS']
        self.metadata['object_path'] = self.metadata['object_path'].apply(lambda x: x.replace('night/',
                                                                                              'night_stand/'))
        self.classes = {item: i for i, item in enumerate(list(self.metadata['class'].unique()))}
        self.noise = noise

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, index):
        y = [0] * len(self.classes)
        y[self.classes[self.metadata['class'][index]]] = 1
        x = read_off_vert(self.data_path + 'ModelNet10/' + self.metadata['object_path'][index])
        y_node = [[1, 0]] * len(x)
        if self.noise:
            min_x = min([min(point) for point in x])
            max_x = max([max(point) for point in x])
            x_noise = [[quick_rand(min_x, max_x) for _ in range(3)] for _ in range(len(x)//10)]
            x += x_noise
            y_node += [[0, 1]] * len(x_noise)
        x = torch.FloatTensor(x).T.unsqueeze(0)
        y = torch.FloatTensor(y).unsqueeze(0)
        y_node = torch.FloatTensor(y_node).T.unsqueeze(0)
        return x, y, y_node


def collate_fn(batch):
    return batch


def read_off(file):
    with open(file, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Invalid file')
        n_vertices, n_faces, _ = (int(num) for num in f.readline().strip().split(' '))
        vertices = tuple([[float(num) for num in f.readline().strip().split(' ')] for _ in range(n_vertices)])
        faces = tuple([[int(num) for num in f.readline().strip().split(' ')] for _ in range(n_faces)])
    return vertices, faces


def read_off_vert(file):
    with open(file, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Invalid file')
        n_vertices, _, _ = (int(num) for num in f.readline().strip().split(' '))
        vertices = [[float(num) for num in f.readline().strip().split(' ')] for _ in range(n_vertices)]
    return vertices


def quick_rand(min_, max_):
    return (max_-min_)*random() + min_


if __name__ == '__main__':
    read_off_vert('data/ModelNet10/bathtub/test/bathtub_0107.off')
    ps = PointDataset('data/')
