import torch
from models.model import MLP
from datasets.mnist import get_dataloader

_, test_loader = get_dataloader(batch_size=1)

model = MLP()
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()
#获取一张图片
images, labels = next(iter(test_loader))
print(f"原始图片 shape: {images.shape}")

images = images.reshape(images.shape[0], -1)
print(f"模型输入 shape: {images.shape}")

with torch.no_grad():
    outputs = model(images)

probabilities = torch.softmax(outputs, dim=1)
print("\n模型预测概率:")

for i, probability in enumerate(probabilities[0]):
    print(f"数字 {i}: {probability.item() * 100:.4f}%")

_, predicted = torch.max(outputs, 1)

print(f"真实标签: {labels.item()}")
print(f"模型预测: {predicted.item()}")
predicted_class = predicted.item()
predicted_score = outputs[0][predicted_class].item()
print( f"\n模型认为这张图片最可能是: " f"{predicted_class}" )
print( f"该类别的原始分数: " f"{predicted_score:.4f}" )