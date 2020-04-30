import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class RegDataSet(Dataset):
    def __init__(self, dataset_root, anno_txt_path, lexicon_path, target_size=(200, 32), characters="'-' + '0123456789'", transform=None):
        super(RegDataSet, self).__init__()
        self.dataset_root = dataset_root
        self.anno_txt_path = anno_txt_path
        self.lexicon_path = lexicon_path
        self.target_size = target_size
        self.height = self.target_size[1]
        self.width = self.target_size[0]
        self.characters = characters
        self.imgs = []
        self.lexicons = []
        self.parse_txt()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path, lexicon_index = self.imgs[item].split()
        lexicon = self.lexicons[int(lexicon_index)].strip()
        img = cv2.imread(os.path.join(self.dataset_root, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = img.shape
        if (img_size[1] / (img_size[0] * 1.0)) < 6.4:
            img_reshape = cv2.resize(img, (int(32.0 / img_size[0] * img_size[1]), self.height))
            mat_ori = np.zeros((self.height, self.width - int(32.0 / img_size[0] * img_size[1]), 3), dtype=np.uint8)
            out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
        else:
            out_img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            out_img = np.asarray(out_img).transpose([1, 0, 2])

        label = [self.characters.find(c) for c in lexicon]
        if self.transform:
            out_img = self.transform(out_img)
        return out_img, label

    def parse_txt(self):
        self.imgs = open(os.path.join(self.dataset_root, self.anno_txt_path), 'r').readlines()
        self.lexicons = open(os.path.join(self.dataset_root, self.lexicon_path), 'r').readlines()
