import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import preprocess
import numpy as np
from sklearn.metrics import accuracy_score
import os
from model_1 import CFSPT

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

batch_size = 64
epochs = 20
num_classes = 3
length = 5120
BatchNorm = True  # 是否批量归一化
number = 200  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.7, 0.2, 0.1]  # 测试集验证集划分比例


def data_pre(path, number):
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
        d_path=path, length=length,
        number=number,
        normal=normal,
        rate=rate,
        enc=False, enc_step=28
    )
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_valid = x_valid.astype('float32')
    y_valid = y_valid.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    x_train = x_train[:, :, np.newaxis]
    x_valid = x_valid[:, :, np.newaxis]
    x_test = x_test[:, :, np.newaxis]
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets.long()).sum().item()

        train_acc = 100 * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.long())

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets.long()).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # 加载最佳模型权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Best validation accuracy: {best_val_acc:.2f}%')

    return model


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets.long()).sum().item()

    test_acc = 100 * test_correct / test_total
    test_loss_avg = test_loss / len(test_loader)

    print(f'Test Loss: {test_loss_avg:.4f}, Test Acc: {test_acc:.2f}%')
    return test_loss_avg, test_acc


if __name__ == '__main__':
    path = r".\Train"
    path_test = r".\Test"

    x_train, x_valid, x_test, y_train, y_valid, y_test = data_pre(path, 200)
    x_train1, x_valid1, x_test1, y_train1, y_valid1, y_test1 = data_pre(path_test, 200)

    print('训练样本维度:', x_train.shape)
    print(x_train.shape[0], '训练样本个数')
    print('验证样本的维度', x_valid.shape)
    print(x_valid.shape[0], '验证样本个数')
    print('测试样本的维度', x_test.shape)
    print(x_test.shape[0], '测试样本个数')

    # 转换为PyTorch张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test1, dtype=torch.float32)  # 使用测试集数据
    y_test_tensor = torch.tensor(y_test1, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_shape = x_train.shape[1:]  # (5120, 1)
    model = CFSPT(input_shape).to(device)

    print("Model architecture:")
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model = train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device)

    # 评估模型
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Final test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")