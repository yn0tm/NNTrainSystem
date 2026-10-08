#===========================
#下载MNIST数据集
#===========================
import os
# 获取当前脚本 mnist_dataset.py 的完整路径
script_path = os.path.abspath(__file__)
# 脚本所在文件夹：NNTrainSystem/datasets
script_dir = os.path.dirname(script_path)
# 往上跳一级，得到 NNTrainSystem
project_root = os.path.dirname(script_dir)
# 拼接：NNTrainSystem/data
data_path = os.path.join(project_root, "data")

from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader
# 数据转换
transform = transforms.Compose([transforms.ToTensor()])
# 加载训练数据
train_dataset = datasets.MNIST(
    root = data_path,
    train = True,
    download = True,
    transform = transform
)

# 加载测试数据
test_dataset = datasets.MNIST(
    root = data_path,
    train = False,
    download = True,
    transform = transform
)

"""
image, label = train_dataset[0]
print(image.shape)
print(label)
"""

# 创建DataLoader
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle=True
)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 64,
    shuffle = False
)

images, labels = next(iter(train_loader))

print("image.shape:", images.shape)
print("labels.shape:", labels.shape)
print(labels[:10])