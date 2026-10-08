from datasets.mnist import get_dataloader

train_loader, test_loader = get_dataloader(batch_size=64)

images, labels = next(iter(train_loader))
print(images.size(1))
print(images.shape[0])
"""
print("训练集 batch:")
print("images shape:", images.shape)
print("labels shape:", labels.shape)

print()

print("测试集 batch 数量:", len(test_loader))
print("训练集 batch 数量:", len(train_loader))
"""