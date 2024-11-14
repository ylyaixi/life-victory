import numpy as np
import random

from torch.utils.data import Dataset
from feeders import tools_a


class Feeder(Dataset):
    def __init__(self, data_path, label_path, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.load_data()
        if partition:
            self.right_arm = np.array([8, 10])
            self.left_arm = np.array([7, 9])
            self.right_leg = np.array([14, 16])
            self.left_leg = np.array([13, 15])
            self.head = np.array([1, 2, 3, 4])
            self.torso = np.array([5, 6, 7, 8])
            self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.right_arm, self.head, self.torso), axis=-1)
            # except for joint no.0

    def load_data(self):
        # data: N C V T M
        self.data = np.load(self.data_path)
        self.label = np.load(self.label_path)
        self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        #if self.split == 'train':
        #    self.data = npz_data['x_train']
        #    self.label = np.where(npz_data['y_train'] > 0)[1]
        #    self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        #elif self.split == 'test':
        #    self.data = npz_data['x_test']
        #    self.label = np.where(npz_data['y_test'] > 0)[1]
        #    self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        #else:
        #    raise NotImplementedError('data split only supports train/test')
        #N, T, _ = self.data.shape
        #self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        if valid_frame_num == 0:
            index = 0
            data_numpy = np.array(self.data[index])
            data_numpy = np.array(data_numpy)
            valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        # data_numpy = tools_a.valid_crop_resize(data_numpy,valid_frame_num,self.p_interval,self.window_size)
        num_people = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)
        # if self.random_rot:
        #     data_numpy = tools_a.random_rot(data_numpy)
        if self.uniform:
            data_numpy, index_t = tools_a.valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval,
                                                           self.window_size, self.thres)
        else:
            data_numpy, index_t = tools_a.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval,
                                                          self.window_size, self.thres)

        if self.split == 'train':
            # intra-instance augmentation
            p = np.random.rand(1)
            if p < self.intra_p:

                if 'a' in self.aug_method:
                    if np.random.rand(1) < 0.5:
                        data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if 'b' in self.aug_method:
                    if num_people == 2:
                        if np.random.rand(1) < 0.5:
                            axis_next = np.random.randint(0, 1)
                            temp = data_numpy.copy()
                            C, T, V, M = data_numpy.shape
                            x_new = np.zeros((C, T, V))
                            temp[:, :, :, axis_next] = x_new
                            data_numpy = temp

                if '1' in self.aug_method:
                    data_numpy = tools_a.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools_a.rotate(data_numpy, p=0.5)
                    # pass
                if '3' in self.aug_method:
                    data_numpy = tools_a.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools_a.spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method:
                    data_numpy, index_t = tools_a.temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools_a.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools_a.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools_a.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools_a.drop_joint(data_numpy, p=0.5)

            # inter-instance augmentation
            elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = self.data[adain_idx]
                data_adain = np.array(data_adain)
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(int)
                data_adain = data_adain[:, t_idx]
                data_numpy = tools_a.skeleton_adain_bone_length(data_numpy, data_adain)

            else:
                data_numpy = data_numpy.copy()

        # modality
        if self.data_type == 'b':
            j2b = tools_a.joint2bone()
            data_numpy = j2b(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = tools_a.to_motion(data_numpy)
        elif self.data_type == 'bm':
            j2b = tools_a.joint2bone()
            data_numpy = j2b(data_numpy)
            data_numpy = tools_a.to_motion(data_numpy)
        else:
            data_numpy = data_numpy.copy()

        if self.partition:
            data_numpy = data_numpy[:, :, self.new_idx]

        return data_numpy, index_t, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
