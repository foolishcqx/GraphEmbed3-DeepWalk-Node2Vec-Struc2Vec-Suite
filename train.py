import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2*num_classes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2*num_classes, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_classifier(epochs, embeddings, labels, train_nodes, test_nodes, num_classes, lr=0.001, device="cuda"):
    # 准备训练和测试数据
    X_train = torch.tensor([embeddings[str(node)] for node in train_nodes], dtype=torch.float32).to(device)
    y_train = torch.tensor([labels[node] for node in train_nodes], dtype=torch.long).to(device)
    X_test = torch.tensor([embeddings[str(node)] for node in test_nodes], dtype=torch.float32).to(device)
    y_test = torch.tensor([labels[node] for node in test_nodes], dtype=torch.long).to(device)

    # 计算类的权重，处理类别不平衡
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=torch.arange(num_classes).numpy(), 
        y=y_train.cpu().numpy()
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # 定义分类器模型
    model = Classifier(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # 类别平衡的交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    acc_list = []
    train_loss = []
    # 开始训练
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        # 打印每个 epoch 的损失
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        if (epoch + 1) % 100 == 0:
            # 测试模型并计算准确率
            model.eval()
            with torch.no_grad():
                predictions = torch.argmax(model(X_test), dim=1)
                acc = accuracy_score(y_test.cpu().numpy(), predictions.cpu().numpy())
                acc_list.append(acc)
            model.train()
    return acc_list, train_loss
