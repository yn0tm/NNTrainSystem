# NNTrainSystem

一个基于 PyTorch 实现的 MNIST 手写数字分类深度学习训练系统。

本项目使用多层感知机（MLP）完成 MNIST 手写数字分类，从数据加载、模型构建、训练、模型保存，到测试和单张图片推理，完整实现了一个基础的深度学习项目流程。

---

## 1. 项目简介

本项目是我的第一个完整深度学习实践项目，主要目的是熟悉 PyTorch 深度学习项目的基本开发流程。

项目主要完成以下功能：

* MNIST 数据集加载
* 数据预处理
* DataLoader 批量数据加载
* MLP 神经网络构建
* 模型训练
* Loss 计算
* Adam 优化器
* 模型参数保存
* 模型参数加载
* 测试集准确率评估
* 单张图片推理
* Softmax 概率分析

通过这个项目，完整实践了一个深度学习模型从训练到推理的基本流程。

---

## 2. 技术栈

* Python 3.10
* PyTorch 2.14.1
* torchvision
* MNIST
* Anaconda / Conda
* VS Code
* Git / GitHub

---

## 3. 项目结构

```text
NNTrainSystem/
│
├── data/
│
├── datasets/
│   └── mnist.py
│
├── models/
│   └── mlp.py
│
├── checkpoints/
│   └── best_model.pth
│
├── train.py
├── test.py
├── predict.py
├── README.md
└── .gitignore
```

### 文件说明

| 文件 / 文件夹            | 作用                    |
| ------------------- | --------------------- |
| `data/`             | 数据集存储目录               |
| `datasets/mnist.py` | MNIST 数据集及 DataLoader |
| `models/mlp.py`     | MLP 模型定义              |
| `checkpoints/`      | 保存训练好的模型参数            |
| `train.py`          | 模型训练                  |
| `test.py`           | 测试集评估                 |
| `predict.py`        | 单张图片推理                |P
| `.gitignore`        | Git 忽略文件配置            |

---

## 4. 模型结构

本项目使用一个简单的多层感知机（MLP）进行 MNIST 分类。

MNIST 图片大小为：

```text
28 × 28
```

因此首先将图片展平为：

```text
784
```

模型结构：

```text
Input
784
 ↓
Linear
784 → 400
 ↓
ReLU
 ↓
Linear
400 → 200
 ↓
ReLU
 ↓
Linear
200 → 10
 ↓
Output
```

对应代码：

```python
self.linear1 = nn.Linear(input_dim, hidden_dim1)
self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
self.linear3 = nn.Linear(hidden_dim2, num_classes)
```

其中：

* 输入维度：784
* 第一隐藏层：400
* 第二隐藏层：200
* 输出类别：10

10 个输出分别对应数字：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

---

## 5. 数据处理

项目使用 `torchvision.datasets.MNIST` 加载 MNIST 数据集。

数据通过 `DataLoader` 按 batch 进行加载。

训练过程中：

```python
images, labels = ...
```

图片原始 shape：

```text
[batch_size, 1, 28, 28]
```

由于 MLP 的第一层需要输入：

```text
[batch_size, 784]
```

因此在进入模型之前进行展平：

```python
images = images.reshape(images.size(0), -1)
```

---

## 6. 模型训练流程

模型训练的基本流程：

```text
读取一个 batch
      ↓
数据预处理
      ↓
Forward
      ↓
计算 Loss
      ↓
optimizer.zero_grad()
      ↓
loss.backward()
      ↓
optimizer.step()
      ↓
下一个 batch
```

核心代码：

```python
outputs = model(images)

loss = criterion(outputs, labels)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

本项目使用：

```python
criterion = nn.CrossEntropyLoss()
```

作为分类任务的损失函数。

优化器使用：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

---

## 7. 模型保存

训练完成后，将模型参数保存到：

```text
checkpoints/best_model.pth
```

之后可以重新创建模型，并通过：

```python
model.load_state_dict(...)
```

加载训练好的参数。

这样就不需要每次使用模型时重新训练。

---

## 8. 模型测试

使用 MNIST 测试集对训练好的模型进行评估。

测试时：

```python
model.eval()
```

并使用：

```python
with torch.no_grad():
```

关闭梯度计算。

最终测试结果：

```text
Test Accuracy: 97.76%
```

模型在 MNIST 测试集上的准确率达到：

**97.76%**

---

## 9. 单张图片推理

项目还实现了单张 MNIST 图片预测。

推理流程：

```text
MNIST 图片
    ↓
[1, 1, 28, 28]
    ↓
展平
    ↓
[1, 784]
    ↓
MLP
    ↓
10 个 Logits
    ↓
Softmax
    ↓
10 个类别概率
    ↓
选择最大概率
    ↓
预测结果
```

例如模型对一张真实标签为 `7` 的图片进行预测：

```text
真实标签: 7
模型预测: 7
```

模型输出的 Logits 中：

```text
数字 7: 18.7018
数字 9: 0.4704
```

其中数字 `7` 的输出最大，因此最终预测结果为：

```text
7
```

进一步通过 Softmax 可以得到各类别的预测概率。

---

## 10. 训练结果

本次训练过程中，Loss 呈现正常下降趋势。

部分训练结果：

```text
Epoch [1/10], Loss: 0.2137
Epoch [2/10], Loss: 0.0883
Epoch [3/10], Loss: 0.0624
Epoch [4/10], Loss: 0.0477
Epoch [5/10], Loss: 0.0407
```

最终测试集准确率：

```text
97.76%
```

---

## 11. 如何运行

### 11.1 创建环境

项目使用 Conda 环境：

```bash
conda create -n nnts_env python=3.10
```

激活环境：

```bash
conda activate nnts_env
```

安装 PyTorch 和 torchvision：

```bash
python -m pip install torch torchvision
```

---

### 11.2 训练模型

运行：

```bash
python train.py
```

训练完成后，模型参数会保存到：

```text
checkpoints/best_model.pth
```

---

### 11.3 测试模型

运行：

```bash
python test.py
```

示例输出：

```text
Test Accuracy: 97.76%
```

---

### 11.4 进行单张图片预测

运行：

```bash
python predict.py
```

可以查看模型对单张 MNIST 图片的预测结果，以及模型输出的类别分数和概率。

---

## 12. 项目中遇到的问题与解决

### 问题 1：`optimizer.zero_grad` 没有真正执行

错误写法：

```python
optimizer.zero_grad
```

正确写法：

```python
optimizer.zero_grad()
```

少了 `()`，实际上只是获取函数对象，并没有调用函数。

由于 PyTorch 默认会累积梯度，如果没有正确清除上一轮的梯度，就可能导致训练过程异常。

这是本项目中遇到的一个重要 Debug 问题。

---

### 问题 2：MLP 输入维度不匹配

MNIST 图片：

```text
[batch_size, 1, 28, 28]
```

而模型第一层：

```python
nn.Linear(784, 400)
```

需要：

```text
[batch_size, 784]
```

因此需要：

```python
images = images.reshape(images.size(0), -1)
```

---

### 问题 3：测试和训练模式的区别

训练时：

```python
model.train()
```

测试和推理时：

```python
model.eval()
```

推理过程中使用：

```python
with torch.no_grad():
```

避免不必要的梯度计算。

---

## 13. 本项目学习到的知识

通过本项目，主要学习和实践了：

### PyTorch 基础

* Tensor
* Dataset
* DataLoader
* `nn.Module`
* `nn.Linear`
* ReLU
* CrossEntropyLoss
* Adam
* Forward
* Backward
* Gradient
* Optimizer

### 深度学习训练流程

* Batch
* Epoch
* Iteration
* Loss
* Accuracy
* 梯度计算
* 参数更新
* 模型保存
* 模型加载

### 模型评估与推理

* `model.train()`
* `model.eval()`
* `torch.no_grad()`
* Logits
* Softmax
* Classification
* Inference

### 工程实践

* Conda 环境管理
* PyTorch 环境配置
* 项目目录结构
* Debug
* Git
* GitHub
* README 编写

---

## 14. 后续计划

这个项目主要用于掌握 PyTorch 深度学习项目的基本流程。

后续将继续学习更加适合图像任务的卷积神经网络（CNN），并逐步实践：

```text
MLP
 ↓
CNN
 ↓
CNN 图像分类项目
 ↓
更复杂的网络结构
 ↓
计算机视觉项目
```

---

## 15. 项目总结

这是我的第一个完整深度学习实践项目。

通过这个项目，我完成了从：

```text
环境配置
 ↓
数据加载
 ↓
数据预处理
 ↓
模型设计
 ↓
模型训练
 ↓
Loss / Accuracy
 ↓
模型保存
 ↓
模型加载
 ↓
测试集评估
 ↓
单张图片推理
```

的完整流程。

项目最终在 MNIST 测试集上取得：

```text
97.76% Test Accuracy
```

后续将在此基础上继续学习 CNN 以及更加复杂的深度学习模型。
