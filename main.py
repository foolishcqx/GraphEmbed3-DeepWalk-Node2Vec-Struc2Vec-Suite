from utils import load_graph, load_labels, split_data
from models.deepwalk import DeepWalk
from models.node2vec import Node2Vec
from models.struc2vec import Struc2Vec
from train import train_classifier
import torch
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

# 设置随机种子以保证实验的可复现性
def set_random_seed(seed=42):
    # 设置 Python 随机数生成器的种子
    random.seed(seed)
    
    # 设置 NumPy 随机数生成器的种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的 CPU 随机种子
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def plot_class_distribution(labels, train_nodes, test_nodes):
    # 获取训练集和测试集的标签
    train_labels = [labels[node] for node in train_nodes]
    test_labels = [labels[node] for node in test_nodes]

    # 统计训练集和测试集的类别分布
    train_class_counts = Counter(train_labels)
    test_class_counts = Counter(test_labels)

    # 提取所有类别
    classes = list(set(labels.values()))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # 创建柱状图
    bar_width = 0.35
    index = np.arange(len(classes))

    # 创建柱状图数据
    train_counts = [train_class_counts.get(cls, 0) for cls in classes]
    test_counts = [test_class_counts.get(cls, 0) for cls in classes]

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    train_bars = ax.bar(index, train_counts, bar_width, label='Train')
    test_bars = ax.bar(index + bar_width, test_counts, bar_width, label='Test')

    # 设置标签和标题
    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequency')
    ax.set_title('Class Distribution in Train and Test Sets')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()

    # 显示每个柱子的数量
    for bars, counts in zip([train_bars, test_bars], [train_counts, test_counts]):
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                    ha='center', va='bottom', fontsize=10)

    # 显示图形
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载数据
    set_random_seed(42)
    graph = load_graph("data/wiki/Wiki_edgelist.txt")
    labels = load_labels("data/wiki/wiki_labels.txt")
    train_nodes, test_nodes = split_data(labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # 参数设置
    algorithm = "struc2vec"  # 可选: "deepwalk", "node2vec", "struc2vec"
    embedding_dim = 512 
    #plot_class_distribution(labels, train_nodes, test_nodes)#获取训练集和测试集的标签分布
    # 图嵌入训练
    if algorithm == "deepwalk":
        model = DeepWalk(graph, embedding_dim, walk_length=20, num_walks=20, window_size=3)
        walks = model.generate_walks()
        embeddings = model.train(walks)
    elif algorithm == "node2vec":
        model = Node2Vec(graph, embedding_dim, walk_length=20, num_walks=20, p=1, q=4, window_size=3)
        walks = model.generate_walks()
        embeddings = model.train(walks)
    elif algorithm == "struc2vec":
        model = Struc2Vec(graph, embedding_dim, walk_length=20, num_walks=20, window_size=3, num_layers=5)
        walks = model.generate_walks()
        embeddings = model.train(walks)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # 节点分类与准确率评估
    num_classes = len(set(labels.values()))
    print(f"Number of classes: {num_classes}")
    num_epochs = 1500
    accuracy, train_loss = train_classifier(num_epochs,embeddings, labels, train_nodes, test_nodes, num_classes, device=device,lr=0.0005)
    print(f"Max Classification Accuracy ({algorithm}): {max(accuracy):.4f}")
    plt.plot(range(1,num_epochs+1), train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss ({algorithm})")
    plt.figure()
    plt.plot(range(0,num_epochs, 100), accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Classification Accuracy ({algorithm})")
    plt.show()