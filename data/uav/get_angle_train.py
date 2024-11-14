import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap
from ThirdOrderRep import getThridOrderRep

def gen_angle_data_from_npy(train_joint_path, train_label_path):
    # 加载数据
    train_x = np.load(train_joint_path)
    train_y = np.load(train_label_path)


    N_train,C, T_train, V, M = train_x.shape

    # 创建内存映射文件
    new_train_x = open_memmap('angle_train_x.npy', dtype='float32', mode='w+', shape=(N_train, 9, T_train, V, 2))


    # 并行处理训练数据
    Parallel(n_jobs=8)(delayed(lambda i: new_train_x.__setitem__(i, getThridOrderRep(train_x[i])))(i) for i in tqdm(range(N_train)))



    # 保存处理后的数据
    np.save("data/train_angle_joint.npy", new_train_x)

if __name__ == '__main__':
    gen_angle_data_from_npy( 'data/train_joint.npy', 'data/train_label.npy')
