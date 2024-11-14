import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

def load_scores(file_path):
    with open(file_path, 'rb') as file:
        return list(pickle.load(file).items())

def calculate_accuracy(labels, fused_results):
    top1_acc = sum(int(np.argmax(result) == int(l)) for result, l in zip(fused_results, labels)) / len(labels)
    top5_acc = sum(int(int(l) in result.argsort()[-5:]) for result, l in zip(fused_results, labels)) / len(labels)
    return top1_acc, top5_acc

if __name__ == "__main__":
    # 定义工作目录路径
    paths = {
        "degcn_val": [
            "ensemble_results/degcn/val/de_joint/weights",
            "ensemble_results/degcn/val/jmf_joint/weights",
            "ensemble_results/degcn/val/jbf_joint/weights",
            "ensemble_results/degcn/val/angle_joint/weights",
        ],
        "degcn_test": [
            "ensemble_results/degcn/test/de_joint/weights",
            "ensemble_results/degcn/test/jmf_joint/weights",
            "ensemble_results/degcn/test/jbf_joint/weights",
            "ensemble_results/degcn/test/angle_joint/weights",
        ],

        "tegcn_val": ["ensemble_results/tegcn/val/tegcn_jb"],
        "tegcn_test": ["ensemble_results/tegcn/test/tegcn_jb"],

        "SkateFormer_val": [
            "ensemble_results/SkateFormer/val/ske_joint",
            "ensemble_results/SkateFormer/val/ske_bone",
        ],
        "SkateFormer_test": [
            "ensemble_results/SkateFormer/test/ske_joint",
            "ensemble_results/SkateFormer/test/ske_bone",
        ],

        "SttFormer_val": [
            "ensemble_results/sttformer/val/joint",
            "ensemble_results/sttformer/val/motion"
        ],
        "SttFormer_test": [
            "ensemble_results/sttformer/test/joint",
            "ensemble_results/sttformer/test/motion"
        ],
    }

    # 加载标签数据
    dataset_type = "test"  # 更改为 "test" 以使用测试集
    label_path = f"data/{dataset_type}_label.npy"
    with open(label_path, 'rb') as f:
        labels = np.load(f)
        print(labels.shape)

    # 加载模型分数文件
    score_files = [load_scores(os.path.join(path, 'epoch1_test_score.pkl'))
                   for path in paths[f'degcn_{dataset_type}'] +
                               paths[f'tegcn_{dataset_type}'] +
                               paths[f'SkateFormer_{dataset_type}'] +
                               paths[f'SttFormer_{dataset_type}']
                   ]

    # 50.85
    optimal_weights = [0.04122027,0.17082286,0.03982683,0.16690768,0.17165557,0.13782612,0.13259714,0.07938541,0.05975812]

    alpha = optimal_weights[:4]
    alpha2 = optimal_weights[4:5]
    alpha3 = optimal_weights[5:7]
    alpha4 = optimal_weights[7:9]

    # 存储融合结果
    fused_results = []

    # 循环计算每个标签的准确率
    for i in tqdm(range(len(labels))):
        label = labels[i]

        # 提取分数
        results1 = [scores[i][1] for scores in score_files[:4]]  # 使用 alpha 权重
        result2 = score_files[4][i][1]  # 使用 alpha2 权重
        results3 = [scores[i][1] for scores in score_files[5:7]]  # 使用 alpha3 权重
        results4 = [scores[i][1] for scores in score_files[7:9]]  # 使用 alpha3 权重


        # 计算加权结果
        result1 = sum(res * w for res, w in zip(results1, alpha))
        result2 = result2 * alpha2[0]
        result3 = sum(res * w for res, w in zip(results3, alpha3))
        result4 = sum(res * w for res, w in zip(results4, alpha4))

        # 融合所有结果
        fused_result = result1 + result2 + result3 + result4
        fused_results.append(fused_result)

    # 转换融合结果为numpy数组并保存
    fused_results_array = np.array(fused_results)
    np.save(f'pred.npy', fused_results_array)

    # 计算并输出Top-1和Top-5准确率
    top1_acc, top5_acc = calculate_accuracy(labels, fused_results_array)
    print(f'Top1 Acc ({dataset_type}): {top1_acc * 100:.4f}%')
    print(f'Top5 Acc ({dataset_type}): {top5_acc * 100:.4f}%')
    print(f'Fusion results saved to pred_{dataset_type}.npy')

