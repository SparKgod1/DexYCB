import os.path
import numpy as np
import re
import shutil


subject_pattern = r"\b\d{8}-subject-\d{2}\b"
seq_pattern = r"\b\d{8}_\d{6}\b"
camera_pattern = r"\b\d{12}\b"
data_pth = "F:\\DexYCB\\lx\\example\\dataset_3d"

for setup in ('s0', 's1', 's2', 's3'):
    for split in ('train', 'val', 'test'):
        name = '{}_{}'.format(setup, split)
        print(f'Dataset name: {name}')
        dataset = np.load("dataset/{}.pkl".format(name), allow_pickle=True)
        dataset_pth = os.path.join(data_pth, name)

        for i, frame in enumerate(dataset):
            # 提取label信息
            color_pth = frame["color_file"]
            depth_pth = frame["depth_file"]
            sub_match = re.search(subject_pattern, color_pth)
            seq_match = re.search(seq_pattern, color_pth)
            camera_match = re.search(camera_pattern, color_pth)
            subject, seq_id, camera_id = sub_match.group(), seq_match.group(), camera_match.group()

            if setup != 's2' and not camera_id == "836212060125":
                continue
            elif split == 'train' and not camera_id == "836212060125":
                continue
            pth = os.path.join(dataset_pth, subject, seq_id)
            shutil.copy(color_pth, pth)
            shutil.copy(depth_pth, pth)
