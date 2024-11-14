import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap
from ThirdOrderRep import getThridOrderRep

def gen_angle_data_from_npy(val_joint_path, val_label_path):
    # 加载数据
    val_x = np.load(val_joint_path)
    val_y = np.load(val_label_path)


    N_val,C, T_val, V, M = val_x.shape

    # 创建内存映射文件
    new_val_x = open_memmap('angle_val_x.npy', dtype='float32', mode='w+', shape=(N_val, 9, T_val, V, 2))


    # 并行处理训练数据
    Parallel(n_jobs=8)(delayed(lambda i: new_val_x.__setitem__(i, getThridOrderRep(val_x[i])))(i) for i in tqdm(range(N_val)))



    # 保存处理后的数据
    np.save("data/val_angle_joint.npy", new_val_x)

if __name__ == '__main__':
    gen_angle_data_from_npy( 'data/val_joint.npy', 'data/val_label.npy')
