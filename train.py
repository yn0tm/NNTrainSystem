import torch
import torch.nn as nn
from datasets.mnist import get_dataloader
from models.model import MLP

train_loader, test_loader = get_dataloader(batch_size=64)
model = MLP()
# 损失函数
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
#训练模式
model.train()

for epoch in range(epochs):
    total_loss = 0

    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.reshape(images.size(0), -1)

        optimizer.zero_grad()
        outputs = model(images)#这个输出结果是一个[64, 10]64张每行10个数据，为对应数字0-9的分数，对应分数最大的认定为那个数字
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        """_, predicted = torch.max(
            outputs.data,
            1
        )"""
        total += labels.size(0)

        correct += (predicted==labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss/len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss:{avg_loss:.4f},"
        f"Accuracy:{accuracy:.2f}%"
    )

#保存训练的参数数据
torch.save(
    model.state_dict(),
    "checkpoints/best_model.pth"
)


"""
import os
os.makedirs("checkpoints", exist_ok=True)
torch.save(
    model.state_dict(),
    "checkpoints/best_model.pth"
)
"""
#torch.save()函数不会创建checkpoints文件夹，但是会穿件best_model.pth文件