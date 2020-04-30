import torch
import cv2
import numpy as np
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch, T=50):
    # T 根据设计的网络输出来
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    labels = list(items[1])
    items[1] = []
    target_lengths = torch.zeros((len(batch,)), dtype=torch.int)
    input_lengths = torch.zeros(len(batch,), dtype=torch.int)
    for idx, label in enumerate(labels):
        target_lengths[idx] = len(label)
        items[1].extend(label)
        input_lengths[idx] = T

    return items[0], torch.tensor(items[1]), target_lengths, input_lengths


def process_img(img, height, width, transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape
    if (img_size[1] / (img_size[0] * 1.0)) < 6.4:
        img_reshape = cv2.resize(img, (int(31.0 / img_size[0] * img_size[1]), height))
        mat_ori = np.zeros((height, width - int(31.0 / img_size[0] * img_size[1]), 3), dtype=np.uint8)
        out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
    else:
        out_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        out_img = np.asarray(out_img).transpose([1, 0, 2])

    out_img = transform(out_img)

    return torch.unsqueeze(out_img, 0)


def decode_out(str_index, characters):
    char_list = []
    for i in range(len(str_index)):
        if str_index[i] != 0 and (not (i > 0 and str_index[i - 1] == str_index[i])):
            char_list.append(characters[str_index[i]])
    return ''.join(char_list)