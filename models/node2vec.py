from gensim.models import Word2Vec
import random

class Node2Vec:
    def __init__(self, graph, embedding_dim, walk_length, num_walks, p, q, window_size):
        self.graph = graph  # 输入的图
        self.embedding_dim = embedding_dim  # 嵌入向量的维度
        self.walk_length = walk_length  # 随机游走的长度
        self.num_walks = num_walks  # 每个节点生成随机游走的次数
        self.p = p  # 返回概率
        self.q = q  # 探索概率
        self.window_size = window_size  # Word2Vec的上下文窗口大小

    def biased_random_walk(self, start_node):
        walk = [start_node]  # 初始化游走路径为起始节点
        for _ in range(self.walk_length - 1):
            cur = walk[-1]  # 当前节点
            neighbors = list(self.graph.neighbors(cur))  # 获取当前节点的邻居节点

            # 如果没有邻居节点，回退到上一节点
            if not neighbors:
                if len(walk) > 1:
                    walk.pop()  # 移除当前节点
                else:
                    break  # 如果是起始节点没有回退余地则结束游走
                continue

            # 如果是第一个节点，直接随机选择邻居
            if len(walk) == 1:
                walk.append(random.choice(neighbors))
            else:
                prev = walk[-2]  # 上一个节点
                probabilities = self.compute_transition_probabilities(prev, cur, neighbors)
                # 根据计算的转移概率随机选择下一个节点
                walk.append(random.choices(neighbors, weights=probabilities, k=1)[0])
        return walk

    def compute_transition_probabilities(self, prev, cur, neighbors):
        probabilities = []
        for neighbor in neighbors:
            if neighbor == prev:
                probabilities.append(1 / self.p)  # 返回节点的概率
            elif neighbor in self.graph[cur]:
                probabilities.append(1)  # 当前节点的邻居
            else:
                probabilities.append(1 / self.q)  # 远离当前节点的邻居
        return probabilities

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())  # 获取图中的所有节点
        for _ in range(self.num_walks):
            random.shuffle(nodes)  # 随机打乱节点顺序
            for node in nodes:
                walks.append(self.biased_random_walk(node))  # 为每个节点生成随机游走
        return walks

    def train(self, walks):
        # 使用Word2Vec对生成的节点序列进行训练
        model = Word2Vec(sentences=walks, vector_size=self.embedding_dim, window=self.window_size, min_count=0, sg=1)
        return {str(node): model.wv[str(node)] for node in self.graph.nodes()}  # 返回节点嵌入向量
