import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap
from ThirdOrderRep import getThridOrderRep

def gen_angle_data_from_npy(test_joint_path, test_label_path):
    # 加载数据
    test_x = np.load(test_joint_path)
    test_y = np.load(test_label_path)


    N_test,C, T_test, V, M = test_x.shape

    # 创建内存映射文件
    new_test_x = open_memmap('angle_test_x.npy', dtype='float32', mode='w+', shape=(N_test, 9, T_test, V, 2))


    # 并行处理训练数据
    Parallel(n_jobs=8)(delayed(lambda i: new_test_x.__setitem__(i, getThridOrderRep(test_x[i])))(i) for i in tqdm(range(N_test)))



    # 保存处理后的数据
    np.save("data/test_angle_joint.npy", new_test_x)

if __name__ == '__main__':
    gen_angle_data_from_npy( 'data/test_joint.npy', 'data/test_label.npy')
