import os.path
import pickle
import yaml
import numpy as np
import re


entri_dict = {
    '20200709-subject-01': "extrinsics_20200702_151821",
    '20200813-subject-02': "extrinsics_20200813_100608",
    '20200820-subject-03': "extrinsics_20200820_091149",
    '20200903-subject-04': "extrinsics_20200903_072753",
    '20200908-subject-05': "extrinsics_20200907_105926",
    '20200918-subject-06': "extrinsics_20200918_092020",
    '20200928-subject-07': "extrinsics_20200928_082347",
    '20201002-subject-08': "extrinsics_20201001_200551",
    '20201015-subject-09': "extrinsics_20201014_215638",
    '20201022-subject-10': "extrinsics_20201022_091549"
}


subject_pattern = r"\b\d{8}-subject-\d{2}\b"
seq_pattern = r"\b\d{8}_\d{6}\b"
camera_pattern = r"\b\d{12}\b"
joint_ind = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
entri_pth = "F:\\DexYCB\\dataset\\dex-ycb-20210415\\calibration"
data_pth = "F:\\DexYCB\\lx\\example\\dataset_3d"
pkl_pth = "F:\\DexYCB\\lx\\example\\dataset"
if not os.path.exists(data_pth):
    os.makedirs(data_pth)
seq_id_old = None
subject_old = None

for setup in ('s0', 's1', 's2', 's3'):
    for split in ('train', 'val', 'test'):
        name = '{}_{}'.format(setup, split)
        print(f'Dataset name: {name}')
        dataset = np.load("{}/{}.pkl".format(pkl_pth, name), allow_pickle=True)
        dataset_pth = os.path.join(data_pth, name)
        if not os.path.exists(dataset_pth):
            os.makedirs(dataset_pth)

        for i, frame in enumerate(dataset):
            # 提取label信息
            label_pth = frame["label_file"]
            frame_label = np.load(label_pth)
            sub_match = re.search(subject_pattern, label_pth)
            seq_match = re.search(seq_pattern, label_pth)
            camera_match = re.search(camera_pattern, label_pth)
            subject, seq_id, camera_id = sub_match.group(), seq_match.group(), camera_match.group()

            if setup != 's2' and not camera_id == "836212060125":
                continue
            elif split == 'train' and not camera_id == "836212060125":
                continue
            subj_pth = os.path.join(dataset_pth, subject)
            if not os.path.exists(subj_pth):
                os.makedirs(subj_pth)

            if subject_old != subject:
                with open(os.path.join(entri_pth, entri_dict[subject], "extrinsics.yml"), 'r') as f:
                    camara_entri = yaml.load(f, Loader=yaml.FullLoader)
                extri_all = camara_entri["extrinsics"]
                subject_old = subject

            extrinsics = extri_all[camera_id]
            # 坐标变换
            joint_3d = np.squeeze(frame_label["joint_3d"])  # (21,3)
            joint_3d = np.array([joint_3d[i] for i in joint_ind])  # (10,3)
            ones = np.ones((joint_3d.shape[0], 1), dtype=joint_3d.dtype)

            trans_matrix = np.array([
                [extrinsics[0], extrinsics[1], extrinsics[2], extrinsics[3]],
                [extrinsics[4], extrinsics[5], extrinsics[6], extrinsics[7]],
                [extrinsics[8], extrinsics[9], extrinsics[10], extrinsics[11]],
                [0, 0, 0, 1]
            ])
            joint_3d = np.hstack((joint_3d, np.ones((joint_3d.shape[0], 1), dtype=joint_3d.dtype)))
            joint_3d = np.dot(trans_matrix, joint_3d.T)
            joint_3d = joint_3d.T
            joint_3d = joint_3d[:, :3]
            joint_3d = np.expand_dims(joint_3d, axis=0)

            # mano参数
            pose_m = frame_label["pose_m"]
            mano_betas = np.array(frame["mano_betas"])
            mano_side = frame["mano_side"]

            # 写入文件
            if seq_id_old:  # 不是第一个序列的第一帧
                if seq_id_old == seq_id:  # 序列没切换
                    joint_3d_world = np.concatenate((joint_3d_world, joint_3d), axis=0)
                    mano_pose = np.concatenate((mano_pose, pose_m), axis=0)
                    if i == len(dataset) - 1:  # 最后一个序列最后一帧，保存
                        np.savez_compressed(os.path.join(seq_pth, "joint_3d_world.npz"), joint_3d_world=joint_3d_world)
                        np.savez_compressed(os.path.join(seq_pth, "mano.npz"), pose=mano_pose, betas=mano_b, side=mano_s)

                else:  # 切换序列
                    # 先保存原本的
                    np.savez_compressed(os.path.join(seq_pth, "joint_3d_world.npz"), joint_3d_world=joint_3d_world)
                    np.savez_compressed(os.path.join(seq_pth, "mano.npz"), pose=mano_pose, betas=mano_b, side=mano_s)
                    # 再给新的赋初值
                    seq_id_old = seq_id
                    joint_3d_world = joint_3d
                    mano_pose = pose_m
                    mano_b = mano_betas
                    mano_s = mano_side
                    seq_pth = os.path.join(subj_pth, seq_id_old)
                    if not os.path.exists(seq_pth):
                        os.makedirs(seq_pth)

            else:  # 第一个序列第一帧
                seq_id_old = seq_id
                joint_3d_world = joint_3d
                mano_pose = pose_m
                mano_b = mano_betas
                mano_s = mano_side
                seq_pth = os.path.join(subj_pth, seq_id_old)
                if not os.path.exists(seq_pth):
                    os.makedirs(seq_pth)

