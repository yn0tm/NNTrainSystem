import torch

from models.model import MLP
from datasets.mnist import get_dataloader

# =========================
# 1. 加载测试数据
# =========================
_, test_loader = get_dataloader(batch_size=64)
# =========================
# 2. 创建模型
# =========================
model = MLP()
# =========================
# 3. 加载训练好的模型参数
# =========================
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
# =========================
# 4. 开始测试
# =========================
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(images.shape[0], -1)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted==labels).sum().item()

accuracy = 100*correct/total

print(f"Test Accuracy: {accuracy:.2f}%")