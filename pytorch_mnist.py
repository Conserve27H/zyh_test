import torch
import torch.utils.data as data_utils
import torchvision.datasets as dataset
import torchvision.transforms as transforms

from CNN import CNN

###############数据加载####################

train_data = dataset.MNIST(
    root = "mnist",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

test_data = dataset.MNIST(
    root = "mnist",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

################分批加载###################
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle= True)

test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle= True)

# print(train_loader)
# print(test_loader)
cnn = CNN()

################损失函数######################
loss_func = torch.nn.CrossEntropyLoss()

#################优化函数####################
optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.01)

#################训练过程########################
# epoch  通常指 一次训练数据全部训练一遍
for epoch in range(10):
    for index, (images,labels) in enumerate(train_loader):
        # print(index)
        # print(images)
        # print(labels)
        #前向传播
        outputs = cnn(images)
        # 传入输出层节点和真实标签来计算损失函数
        loss = loss_func(outputs, labels)

        # 先清空梯度
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        optimizer.step()
        # print(outputs)
        # print(outputs.size())
        print("当前为第{}轮，当前批次为{}/{}, loss为{}".format(epoch+1,index+1,len(train_data)//64, loss.item()))

    #######################测试集验证#####################
    loss_test = 0
    rightValue = 0
    for index2,(images, labels) in enumerate(test_loader):
        outputs = cnn(images)
        # print(outputs)
        # print(outputs.size())
        # print(labels)
        # print(labels.size())
        loss_test += loss_func(outputs,labels)

        _, pred = outputs.max(1)
        print(pred)
        # eq(), 把两个张量中的每一个元素进行对比，如果相等，对应位置为True，如果不相等，对应位置为False，返回一个张量
        rightValue += (pred==labels).sum().item()

        print("当前为第{}轮测试集验证，当前批次为{}/{}, loss为{}，准确率是{}".format(epoch + 1,index2 + 1 ,len(test_data)//64,loss_test,rightValue/len(test_data)))

torch.save(cnn, "model/mnist_model.pkl")

