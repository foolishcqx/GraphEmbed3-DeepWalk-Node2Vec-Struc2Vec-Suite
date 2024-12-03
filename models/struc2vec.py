import networkx as nx
from gensim.models import Word2Vec
import itertools
import random
from collections import defaultdict

class Struc2Vec:
    def __init__(self, graph, embedding_dim, walk_length, num_walks, window_size, num_layers=3):
        self.graph = graph  # 输入的图
        self.embedding_dim = embedding_dim  # 嵌入向量的维度
        self.walk_length = walk_length  # 随机游走的长度
        self.num_walks = num_walks  # 每个节点生成随机游走的次数
        self.num_layers = num_layers  # 结构相似性计算的层数
        self.window_size = window_size  # Word2Vec的上下文窗口大小
        self.structural_similarity = defaultdict(dict)  # 结构相似性字典

    def compute_structural_similarity(self):
        # 计算每个节点在每个层次上的结构相似性
        for layer in range(self.num_layers):
            for node in self.graph.nodes():
                self.structural_similarity[layer][node] = []
                neighbors = nx.ego_graph(self.graph, node, radius=layer)
                self.structural_similarity[layer][node] = len(neighbors)

    def generate_walks(self):
        self.compute_structural_similarity()  # 计算结构相似性
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)  # 随机打乱节点顺序
            for node in nodes:
                walks.append(self._struc_walk(node))  # 为每个节点生成结构化游走
        return walks

    def _struc_walk(self, start_node):
        walk = [start_node]
        for _ in range(self.walk_length - 1):
            cur_node = walk[-1]
            neighbors = list(self.graph.neighbors(cur_node))  # 获取当前节点的邻居
            if not neighbors:
                break
            similarity_scores = [self._similarity_score(cur_node, n) for n in neighbors]  # 计算每个邻居的相似性分数
            next_node = random.choices(neighbors, weights=similarity_scores, k=1)[0]  # 根据相似性得分加权选择下一个节点
            walk.append(next_node)
        return walk

    def _similarity_score(self, node1, node2):
        # 计算节点1和节点2的结构相似性分数
        score = 0
        for layer in range(self.num_layers):
            if node2 in self.structural_similarity[layer]:
                score += 1 / abs(self.structural_similarity[layer][node1] - self.structural_similarity[layer][node2] + 1e-5)
        return score

    def train(self, walks):
        # 使用Word2Vec训练节点序列
        model = Word2Vec(sentences=walks, vector_size=self.embedding_dim, window=self.window_size, min_count=0, sg=1)
        return {str(node): model.wv[str(node)] for node in self.graph.nodes()}  # 返回节点嵌入向量
