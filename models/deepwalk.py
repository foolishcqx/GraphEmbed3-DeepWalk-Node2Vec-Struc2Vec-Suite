import torch
import random
from collections import defaultdict
from gensim.models import Word2Vec

class DeepWalk:
    def __init__(self, graph, embedding_dim, walk_length, num_walks, window_size):
        self.graph = graph  # 输入图
        self.embedding_dim = embedding_dim  # 嵌入向量的维度
        self.walk_length = walk_length  # 每次随机游走的长度
        self.num_walks = num_walks  # 每个节点生成随机游走的次数
        self.window_size = window_size  # Word2Vec的上下文窗口大小

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())  # 获取图中所有节点
        for _ in range(self.num_walks):  # 为每个节点生成多个随机游走
            random.shuffle(nodes)  # 随机打乱节点顺序
            for node in nodes:
                walks.append(self.random_walk(node))  # 对每个节点生成随机游走
        return walks

    def random_walk(self, start_node):
        walk = [start_node]  # 游走路径初始化为当前节点
        for _ in range(self.walk_length - 1):  # 游走长度减去一个起始节点
            neighbors = list(self.graph.neighbors(walk[-1]))  # 获取当前节点的邻居节点
            if neighbors:  # 如果有邻居节点
                walk.append(random.choice(neighbors))  # 随机选择一个邻居节点加入路径
            else:
                break  # 如果没有邻居，结束游走
        return walk

    def train(self, walks):
        model = Word2Vec(sentences=walks, vector_size=self.embedding_dim, window=self.window_size, min_count=0, sg=1)
        return {str(node): model.wv[str(node)] for node in self.graph.nodes()}  # 返回节点嵌入向量
