## 一、环境配置

```
Ubuntu22.04
python=3.12
torch=2.4.1
```

> 安装如下安装包

```
pip install -r requirements.txt
```



## 二、数据位置

> 数据存放于**data**目录下，将数据集解压到data目录下

### 数据处理

1. 将数据集解压到data目录下
2. 数据集处理出bone模态数据：运行`python gen_modal.py --modal bone`得到bone模态数据
3. 数据集处理出motion模态数据：运行`python gen_modal.py --modal motion`得到motion模态的数据
4. bone模态与joint模态合并：运行`python gen_modal.py --modal jmb`得到合并模态的数据
5. 将gen_modal.py文件第16行替换为`sets = {'train', 'val'}`
6. 重复3，4,  5步骤
7. 依次运行以下命令

```
python data/uav/get_angle_train.py
python data/uav/get_angle_test.py
python data/uav/get_angle_val.py

python data/uav/get_data.py --train_data_path ./data/train_joint.npy --test_data_path ./data/val_joint.npy --train_label_path ./data/train_label.npy --test_label_path ./data/val_label.npy --save_path ./data/UAV2_joint.npz

python data/uav/get_data_B.py --train_data_path ./data/train_joint.npy --test_data_path ./data/test_joint.npy --train_label_path ./data/train_label.npy --test_label_path ./data/test_label.npy --save_path ./data/UAV2B_joint.npz
```



## 三、训练

### DeGCN:

```
cd DeGCN
python main.py --config config/train/joint.yaml
python main.py --config config/train/jbf.yaml
python main.py --config config/train/jmf.yaml
python main_a.py --config config/train/angle_joint.yaml
```

### TeGCN:

```
cd TeGCN
python main.py --config ./config/train/tegcn_jb.yaml
```

### SkateFormer:

```
cd SkateFormer-main
python main.py --config ./config/uav/uav_j.yaml
python main.py --config ./config/uav/uav_b.yaml 
```

### SttFormer:

```
cd sttformer
python main.py --config ./config/train/joint.yaml --work_dir ./work_dir/train/joint
python main.py --config ./config/train/motion.yaml --work_dir ./work_dir/train/motion
```



## 四、val验证集

> 分别将训练得到的最好的权重放到work_dir的best中对验证集集进行测试。

### DeGCN:

```
cd DeGCN
python main.py --phase test --weights work_dir/best/joint/epoch_69_55752.pt --save-score true --config config/val/joint.yaml
python main.py --phase test --weights work_dir/best/jbf/epoch_73_50808.pt --save-score true --config config/val/jbf.yaml 
python main.py --phase test --weights work_dir/best/jmf/epoch_73_50808.pt --save-score true --config config/val/jmf.yaml
python main_a.py --phase test --weights work_dir/best/angle_j/epoch_70_33390.pt --save-score true --config config/val/angle_joint.yaml
```

### Tegcn:

```
python main.py --phase test --save-score true --config config/val/tegcn_jb.yaml
```

### SkateFormer:

```
cd SkateFormer-main
python main.py --config ./config/val/uav_j.yaml
python main.py --config ./config/val/uav_b.yaml
```

### SttFormer:

```
cd sttformer
python main.py  --weights ./work_dir/best/joint/joint.pt --save_score True --config ./config/val/joint.yaml --work_dir ../ensemble_results/sttformer/val/joint
python main.py  --weights ./work_dir/best/motion/motion.pt --save_score True --config ./config/val/motion.yaml --work_dir ../ensemble_results/sttformer/val/motion
```



## 五、test测试集

> 分别将训练得到的最好的权重放到work_dir的best中对测试集进行测试。

### DeGCN:

```
python main.py --phase test --weights work_dir/best/joint/epoch_69_55752.pt --save-score true --config config/test/joint.yaml
python main.py --phase test --weights work_dir/best/jbf/epoch_73_50808.pt --save-score true --config config/test/jbf.yaml
python main.py --phase test --weights work_dir/best/jmf/epoch_73_50808.pt --save-score true --config config/test/jmf.yaml
python main_a.py --phase test --weights work_dir/best/angle_j/epoch_70_33390.pt --save-score true --config config/test/angle_joint.yaml
```

### Tegcn:

```
python main.py --phase test --save-score true --config config/test/tegcn_jb.yaml
```

### SkateFormer:

```
cd SkateFormer-main
python main.py --config ./config/test/uav_j.yaml
python main.py --config ./config/test/uav_b.yaml
```

### SttFormer:

```
cd sttformer
python main.py --weights ./work_dir/best/joint/joint.pt --save_score True --config ./config/test/joint.yaml --work_dir ../ensemble_results/sttformer/test/joint
python main.py --weights ./work_dir/best/motion/motion.pt --save_score True --config ./config/test/motion.yaml --work_dir ../ensemble_results/sttformer/test/motion
```



## 六、九模态进行Score融合

```
python ensembleall.py
```

得到最终的置信度文件**pred.npy**

自动搜索参数：

1. 运行`python search.py`即可在验证集上搜索最优参数组合（每次得到的最优组合可能不同）
2. 将搜素到并保存在log.txt里的参数放入到ensembleall.py中运行即可得到**pred.npy**