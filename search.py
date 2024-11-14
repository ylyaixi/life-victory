import pickle
import os
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple
from bayes_opt import BayesianOptimization

@dataclass
class ModalityConfig:
    """模态配置类"""
    name: str
    path: str
    data: List = None  # 用于存储加载的数据

class EnsembleEvaluator:
    def __init__(self, modality_configs: List[ModalityConfig], label_path: str):
        self.modality_configs = modality_configs
        self.num_modalities = len(modality_configs)
        self.label = self._load_label(label_path)
        self._load_all_modalities()
        self.all_results = []  # 存储所有的评估结果
        self.best_acc = 0  # 跟踪最佳准确率

    def _load_label(self, label_path: str) -> np.ndarray:
        """加载标签数据"""
        with open(label_path, 'rb') as f:
            return np.load(f)

    def _load_all_modalities(self):
        """加载所有模态的数据"""
        for config in self.modality_configs:
            with open(os.path.join(config.path, 'epoch1_test_score.pkl'), 'rb') as f:
                config.data = list(pickle.load(f).items())

    def evaluate(self, alpha: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """评估函数"""
        # 确保输入的 alpha 是四位小数的归一化值
        alpha = np.round(alpha / np.sum(alpha), decimals=4)

        right_num = right_num_5 = total_num = 0
        fused_results = []

        for i in range(len(self.label)):
            predictions = [config.data[i][1] for config in self.modality_configs]
            result = sum(pred * w for pred, w in zip(predictions, alpha))
            fused_results.append(result)

            l = self.label[i]
            rank_5 = result.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r_max = np.argmax(result)
            right_num += int(r_max == int(l))
            total_num += 1

        acc = right_num / total_num
        acc5 = right_num_5 / total_num

        # 如果当前准确率是最佳的，保存到pred.npy
        if acc > self.best_acc:
            self.best_acc = acc
            np.save('pred.npy', np.array(fused_results))
            print(f"New best accuracy: {acc:.4f}. Results saved to pred.npy")

        # 存储四位小数的归一化结果
        self.all_results.append((alpha, acc, acc5))

        return acc, acc5, np.array(fused_results)

    def optimize_weights_bayesian(self, init_points=5, n_iter=50):
        """使用贝叶斯优化寻找最优权重"""
        def objective(**kwargs):
            alpha = np.array([kwargs[f'alpha_{i}'] for i in range(self.num_modalities)])
            # 在优化过程中就使用归一化的权重
            alpha = np.round(alpha / np.sum(alpha), decimals=4)
            acc, _, _ = self.evaluate(alpha)
            return acc

        pbounds = {f'alpha_{i}': (0, 1) for i in range(self.num_modalities)}

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=1,
        )

        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        best_alpha = np.array([optimizer.max['params'][f'alpha_{i}']
                               for i in range(self.num_modalities)])
        best_alpha = best_alpha / np.sum(best_alpha)
        best_acc, best_acc5, best_fused_results = self.evaluate(best_alpha)

        return best_alpha, best_acc, best_acc5, best_fused_results

def main():
    modality_configs = [
        ModalityConfig("de_J_Score", "ensemble_results/degcn/val/de_joint/weights"),                           # 1
        ModalityConfig("de_JMf_Score", "ensemble_results/degcn/val/jmf_joint"),                        # 6
        ModalityConfig("de_JBMf_Score", "ensemble_results/degcn/val/jbmf_joint/weights"),                      # 8
        ModalityConfig("de_A_Score", "ensemble_results/degcn/val/angle_joint/weights"),
        ModalityConfig("Te_Score", "ensemble_results/tegcn/val/tegcn_jb"),
        ModalityConfig("ske_J_Score", "ensemble_results/SkateFormer/val/ske_joint"),
        ModalityConfig("ske_B_Score", "ensemble_results/SkateFormer/val/ske_bone"),
        ModalityConfig("stt_J_Score", "ensemble_results/sttformer/val/joint"),                                 # s1                             # s2
        ModalityConfig("stt_JM_Score", "ensemble_results/sttformer/val/motion"),                               # s3
    ]

    evaluator = EnsembleEvaluator(modality_configs, 'data/val_label.npy')

    print("Starting Bayesian optimization...")
    best_alpha, best_acc, best_acc5, best_fused_results = evaluator.optimize_weights_bayesian(
        init_points=5,
        n_iter=300  # 增加迭代次数以获得更好的结果
    )

    # 将输出保存到log.txt
    with open('log.txt', 'w') as f:
        # 保存完整精度的归一化权重
        f.write(f'Best alpha (normalized): {best_alpha}\n')
        f.write(f'Best Top1 Acc: {best_acc:.4f}\n')
        f.write(f'Corresponding Top5 Acc: {best_acc5:.4f}\n\n')

        f.write('Results meeting the criteria (acc > 0.51 or acc5 > 0.69):\n')
        for alpha, acc, acc5 in evaluator.all_results:
            if acc > 0.51 or acc5 > 0.69:  # 修改了条件以匹配注释
                f.write(f'Alpha (normalized): {alpha}\n')
                f.write(f'Acc: {acc:.4f}, Acc5: {acc5:.4f}\n\n')

    # 最佳结果已经在evaluate函数中保存，这里只需要打印信息
    print('Best fusion results saved to pred.npy')
    print('Log saved to log.txt')

if __name__ == "__main__":
    main()
