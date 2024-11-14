import os

import numpy as np
from typing import Tuple, List

def load_your_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载训练和测试数据

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, y_train, x_test, y_test
    """
    try:
        x_train = np.load('./data/train_joint.npy')
        x_test = np.load('./data/val_joint.npy')
        y_train = np.load('./data/train_label.npy')
        y_test = np.load('./data/val_label.npy')
        return x_train, y_train, x_test, y_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise

def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 155))
    for idx, l in enumerate(labels):
        labels_vector[int(idx), int(l)] = 1
    return labels_vector
# labels_vector=np.zeros((4999,155))
# labels_vector[:,0]=1

def seq_translation(skes_joints: np.ndarray) -> np.ndarray:
    """
    Args:
        skes_joints (np.ndarray): 输入骨架数据，shape: (N, C, T, V, M)

    Returns:
        np.ndarray: 平移后的骨架数据
    """
    N, C, T, V, M = skes_joints.shape
    skes_joints = skes_joints.copy()

    for n in range(N):
        for m in range(M):
            # Find the first non-zero frame
            first_nonzero_frame = np.argmax(np.any(skes_joints[n, :, :, :, m] != 0, axis=(0, 2)))

            if first_nonzero_frame == 0 and np.all(skes_joints[n, :, 0, :, m] == 0):
                # If the sequence is all zeros, skip it
                continue

            # Set new origin as the second joint (index 1) of the first non-zero frame
            origin = skes_joints[n, :, first_nonzero_frame, 1, m]

            # Perform translation
            for t in range(T):
                skes_joints[n, :, t, :, m] -= origin[:, np.newaxis]

    return skes_joints

    # 将结果转换为numpy数组
    result = np.array(non_zero_samples, dtype=object)
    result_labels = np.array(non_zero_labels)

    print("Output shape from remove_zero_samples_and_augment:", result.shape)
    return result, result_labels



def normalize_skeletons(skes_joints: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    归一化骨架数据

    Args:
        skes_joints (np.ndarray): 输入骨架数据
        epsilon (float): 小的常数，用于避免除以零

    Returns:
        np.ndarray: 归一化后的骨架数据
    """
    print("Input shape to normalize_skeletons:", skes_joints.shape)
    N, C, T, V, M = skes_joints.shape
    for n in range(N):
        for m in range(M):
            # 计算中心，并确保其形状正确
            center = np.mean(skes_joints[n, :, :, 1, m], axis=1)  # shape: (C,)
            center = center[:, np.newaxis, np.newaxis]  # shape: (C, 1, 1)

            # 减去中心
            skes_joints[n, :, :, :, m] -= center

            # 归一化
            max_val = np.max(np.abs(skes_joints[n, :, :, :, m]))
            if max_val > epsilon:
                skes_joints[n, :, :, :, m] /= max_val

    print("Output shape from normalize_skeletons:", skes_joints.shape)
    return skes_joints

import numpy as np
from collections import Counter

def oversample_minority_classes(x_train, y_train, target_samples_per_class=120):
    """
    对样本数量少于目标数量的类别进行过采样

    Args:
        x_train (np.ndarray): 训练数据
        y_train (np.ndarray): 训练标签
        target_samples_per_class (int): 每个类别的目标样本数

    Returns:
        Tuple[np.ndarray, np.ndarray]: 过采样后的训练数据和标签
    """
    class_counts = Counter(y_train)
    oversampled_x = []
    oversampled_y = []

    for class_label, count in class_counts.items():
        # 获取当前类别的所有样本
        class_samples = x_train[y_train == class_label]

        if count < target_samples_per_class:
            # 需要过采样的数量
            oversample_count = target_samples_per_class - count

            # 随机选择样本进行复制
            indices = np.random.choice(count, oversample_count, replace=True)
            additional_samples = class_samples[indices]

            # 合并原始样本和过采样的样本
            oversampled_x.extend(class_samples)
            oversampled_x.extend(additional_samples)
            oversampled_y.extend([class_label] * target_samples_per_class)
        else:
            # 对于样本数量已经足够的类别，直接添加所有样本
            oversampled_x.extend(class_samples)
            oversampled_y.extend([class_label] * count)

    return np.array(oversampled_x), np.array(oversampled_y)

def process_data():
    # 加载数据
    x_train, y_train, x_test, y_test = load_your_data()

    # 原有的预处理步骤
    x_train = seq_translation(x_train)
    # x_train, y_train = remove_zero_samples_and_augment(x_train, y_train)
    x_train = normalize_skeletons(x_train)

    # 对少数类进行过采样
    x_train, y_train = oversample_minority_classes(x_train, y_train, target_samples_per_class=125)

    # 处理测试数据
    x_test = seq_translation(x_test)
    # x_test, y_test = remove_zero_samples_and_augment(x_test, y_test)
    x_test = normalize_skeletons(x_test)

    # 转换标签为one-hot编码
    y_train = one_hot_vector(y_train)
    y_test = one_hot_vector(y_test)

    # 保存处理后的数据
    data_dict = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }
    np.savez('./data/UAV2_joint.npz', **data_dict)
    print("数据处理完成并保存为 UAV2_joint.npz")


# 运行处理函数
if __name__ == "__main__":
    process_data()
"""

# 验证保存的数据
def verify_data():
    data = np.load('UAV2.npz')
    print("加载的数据包含以下键：", data.files)
    print("x_train 形状：", data['x_train'].shape)
    print("y_train 形状：", data['y_train'].shape)
    print("x_test 形状：", data['x_test'].shape)
    print("y_test 形状：", data['y_test'].shape)

    # 检查y_train的详细分类
    y_train = data['y_train']
    y_test = data['y_test']

    # 如果y_train和y_test是one-hot编码，我们需要转换它们
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    print("\ny_train 详细信息:")
    print("最小值:", y_train.min())
    print("最大值:", y_train.max())
    print("唯一值:", np.unique(y_train))
    print("各类别数量:")
    for i in range(155):  # 假设有155个类别
        count = np.sum(y_train == i)
        if count > 0:
            print(f"类别 {i}: {count}")

    print("\ny_test 详细信息:")
    print("最小值:", y_test.min())
    print("最大值:", y_test.max())
    print("唯一值:", np.unique(y_test))
    print("各类别数量:")
    for i in range(155):  # 假设有155个类别
        count = np.sum(y_test == i)
        if count > 0:
            print(f"类别 {i}: {count}")

# 运行验证函数
if __name__ == "__main__":
    verify_data()
"""
