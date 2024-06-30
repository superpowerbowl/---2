import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载并预处理CIFAR-100数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT期望的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

trainset = torchvision.datasets.CIFAR100(root='/root/transformer_vs_CNN-CIFAR100/datasets/cifar-100-python', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR100(root='/root/transformer_vs_CNN-CIFAR100/datasets/cifar-100-python', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)

# 3. 定义ViT模型
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.heads[0] = nn.Linear(model.heads[0].in_features, 100)  # 修改分类头为100类

# 如果有可用的GPU，则将模型转到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_list = []
train_acc_list = []  # 新增训练准确率列表

# 5. 训练模型
for epoch in range(30):  # 遍历数据集多次
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total  # 计算训练准确率
        if i % 200 == 199:  # 每200个批次打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}, train_acc: {train_acc:.2f}%')
            running_loss = 0.0
            train_acc_list.append(train_acc)
            loss_list.append(running_loss / 200)
# 保存模型权重
torch.save(model.state_dict(), 'vit_model_weights.pth')
print('Model weights saved to vit_model_weights.pth')

# 使用 matplotlib 展示训练准确率和损失函数
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_acc_list)
plt.xlabel('Iteration')
plt.ylabel('Training Accuracy (%)')
plt.title('Training Accuracy over Iterations')

plt.subplot(1, 2, 2)
plt.plot(loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')

plt.tight_layout()
plt.show()
plt.savefig('train_transformer.png')

# 6. 评估模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

