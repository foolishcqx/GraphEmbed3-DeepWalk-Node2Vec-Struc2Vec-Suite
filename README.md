# 图嵌入任务实验

## 实验概述

本实验旨在实现和评估图嵌入算法在节点分类任务上的性能。实验基于PyTorch框架，选择了DeepWalk、Node2Vec和Struc2Vec三种图嵌入算法，并在Wiki图数据集上进行实验。

## 数据集描述

数据集为Wiki图，一个有向无权图。包含两个文件：

- `Wiki_edgelist.txt`：描述图中所有边的连接关系。
- `wiki_labels.txt`：包含每个节点的真实类别。

实验中随机选择90%的节点用于分类模块的训练，剩余10%节点用于测试。

## 实验环境

- 编程语言：Python
- 框架：PyTorch
- 库：NetworkX, NumPy, Gensim

## 实验步骤

### 1. 数据加载

使用`load_graph`函数加载图结构，`load_labels`函数加载节点标签。

### 2. 数据分割

使用`split_data`函数将数据集分为训练集和测试集。

### 3. 图嵌入

实现三种图嵌入算法：

- DeepWalk：基于随机游走和Word2Vec模型。
- Node2Vec：在DeepWalk基础上引入有偏随机游走。
- Struc2Vec：基于节点的多层次结构相似性生成随机游走。

### 4. 节点分类

构建简单的神经网络分类器，使用训练得到的节点嵌入向量进行分类任务。

### 5. 训练与评估

训练分类器，并使用训练集和测试集的分类准确率来评估图嵌入算法的效果。

### 6. 结果可视化

对训练集和测试集的类别分布进行可视化，并绘制训练损失曲线。

## 代码结构

```plaintext
- models
	- deepwalk.py
	- node2vec.py
	- struc2vec.py
- utils.py
- train.py
- main.py
- /data
  - wiki
    - Wiki_edgelist.txt
    - wiki_labels.txt
- README.md
```

## 代码说明

```json
models文件夹下包含了DeepWalk、Node2Vec和Struc2Vec三种图嵌入算法的实现

train.py包含了训练脚本

utils.py下包含了数据划分的函数

main.py包含了多个模型的训练以及相应的参数选择

如果你想训练模型，只需要在main.py中选择对应的算法以及修改相应的参数即可
```

