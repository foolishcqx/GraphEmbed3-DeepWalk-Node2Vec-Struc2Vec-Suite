import networkx as nx
import numpy as np
import torch
import random

def load_graph(edge_file):
    graph = nx.read_edgelist(edge_file,create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    return graph

def load_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            node, label = map(int, line.strip().split())
            labels[node] = label
    return labels

def split_data(labels, train_ratio=0.9):
    nodes = list(labels.keys())
    np.random.shuffle(nodes)
    train_size = int(len(nodes) * train_ratio)
    train_nodes = nodes[:train_size]
    test_nodes = nodes[train_size:]
    return train_nodes, test_nodes

def generate_edge_list(graph):
    """
    从 NetworkX 图生成边列表
    """
    edges = list(graph.edges())
    edge_weights = [1] * len(edges)  # 假设所有边的权重为1
    return edges, edge_weights