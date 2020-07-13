from torch.utils.data.dataset import Dataset
from pathlib import Path
import cv2
import random
import numpy as np

class Dataset(Dataset):
    def __init__(self, data_root, sub_id_set, condition_set, view_set):
        super(Dataset, self).__init__()
        self.img_paths = []
        self.sub_ids = []
        self.conditions = []
        self.views = []
        for sub_id in sub_id_set:
            for condition in condition_set:
                for view in view_set:
                    file_path = Path(data_root, f'{sub_id}/{condition}/{sub_id}-{condition}-{view}.png')
                    if file_path.exists():
                        self.img_paths.append(str(file_path))
                        self.sub_ids.append(sub_id)
                        self.conditions.append(condition)
                        self.views.append(view)

    def __getitem__(self, index):
        file_path = self.img_paths[index]
        image = load_image(file_path)
        sub_id = self.sub_ids[index]
        condition = self.conditions[index]
        view = self.views[index]
        return image, sub_id, condition, view

    def __len__(self):
        return len(self.img_paths)

class TripletDataset(Dataset):
    def __init__(self, data_root, sub_id_set, condition_set, view_set):
        self.data_root = data_root
        self.img_paths = []
        self.sub_ids = []
        self.conditions = []
        self.views = []

        for sub_id in sub_id_set:
            for condition in condition_set:
                for view in view_set:
                    file_path = Path(data_root, f'{sub_id}/{condition}/{sub_id}-{condition}-{view}.png')
                    if file_path.exists():
                        self.img_paths.append(str(file_path))
                        self.sub_ids.append(sub_id)
                        self.conditions.append(condition)
                        self.views.append(view)

        self.sub_id_set = set(sub_id_set)
        self.sub_id_to_indices = dict()
        for sub_id in self.sub_id_set:
            id_list = np.where(np.asarray(self.sub_ids) == sub_id)[0]
            self.sub_id_to_indices[sub_id] = id_list

    def __getitem__(self, index):
        img1_path = self.img_paths[index]
        sub_id1 = self.sub_ids[index]
        condition1 = self.conditions[index]
        view1 = self.views[index]

        # generate positive image
        positive_index = index
        # sample the sample subject but different index
        while positive_index == index:
            pos_index_candidates = list(self.sub_id_to_indices[sub_id1])
            positive_index = np.random.choice(pos_index_candidates)

        # generate negative image
        neg_sub_id_candidates = list(self.sub_id_set - set([sub_id1]))
        neg_sub_id = np.random.choice(neg_sub_id_candidates)
        neg_index_candidates = self.sub_id_to_indices[neg_sub_id]
        negative_index = np.random.choice(neg_index_candidates)

        img2_path = self.img_paths[positive_index]
        sub_id2 = self.sub_ids[positive_index]
        condition2 = self.conditions[positive_index]
        view2 = self.views[positive_index]

        img3_path = self.img_paths[negative_index]
        sub_id3 = self.sub_ids[negative_index]
        condition3 = self.conditions[negative_index]
        view3 = self.views[negative_index]

        img1 = load_image(path=img1_path)
        img2 = load_image(path=img2_path)
        img3 = load_image(path=img3_path)
        return img1, img2, img3

    def __len__(self):
        return len(self.img_paths)

def load_image(path, height=64, width=64, channels=1):
    if channels == 3:
        flag = 1
    else:
        flag = 0
    in_image = cv2.imread(path, flag)

    info = np.iinfo(in_image.dtype)
    # convert to [0, 1]
    in_image = in_image.astype(np.float) / info.max

    iw = in_image.shape[1]
    ih = in_image.shape[0]
    if iw < ih:
        in_image = cv2.resize(in_image, (width, int(width * ih/iw)))
    else:
        in_image = cv2.resize(in_image, (int(height * iw / ih), height))
    in_image = in_image[0:width, 0:height]
    # convert to [-1, 1]
    # in_image = 2 * in_image - 1
    if channels == 1:
        in_image = np.expand_dims(in_image, axis=2)
    in_image = np.transpose(in_image, (2, 0, 1))
    return in_image

def sample_image_path(data_root, sub_id, condition_set, view_set):
    file_path = None
    condition = None
    view = None

    while file_path is None:
        condition = random.choice(condition_set)
        view = random.choice(view_set)
        file_path = Path(data_root, f'{sub_id}/{condition}/{sub_id}-{condition}-{view}.png')
        if file_path.exists():
            file_path = str(file_path)
        else:
            file_path = None
    return file_path, condition, view