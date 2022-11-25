# 本代码的作用是将MNIST手写数据集中手写体的数字0~9给正确的识别出来。

'''
pytorch进行神经网络训练的流程：

一.首先定义神经网络模型 model=Net()

二.定义优化器  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

三.训练阶段，循环每个epoch，对于一个epoch：
    1.将模型设置为训练模式  model.train()
    2.将训练数据分为batch_size大小的组，对每一组中所有的样本：
        1).将梯度初始化为0  optimizer.zero_grad()
        2).对这组样本进行训练，并得到输出  output=model(data)
        3).根据预测输出和真实类别计算损失函数  loss=F.nll_loss(output,target)
        4).反向传播，计算梯度  loss.backward()
        5).更新模型参数  optimizer.step()
    3.保存模型（可省略）  torch.save(model,"mnist_torch.pkl")

四.加载已训练好的模型（可省略）

五.测试阶段，对于所有的测试样本：
    1.将模型设置为评价模式  model.eval()
    2.根据建好的模型对测试数据进行预测，得到预测结果  output=model(data)
    3.计算损失函数  test_loss+=F.nll_loss(output,target).data
    4.将概率最大的类别当作预测类别，并计算准确率  correct+=pred.eq(target.data).cpu().sum()

上述第四大步的第4小步是当问题是分类问题时才有的

上述中 model.train()和model.eval()的不同在于，model.eval()去除神经网络中的随机性，而model.train()
则保留了神经网络中的随机性。这是因为神经网络中的某些操作（如dropout等），会存在一定的随机性（dropout
会随机让神经网络中的节点失活），所以为了保证预测结果是可复现的，所以在测试阶段要设置为model.eval()。

'''
import os
import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from keras.datasets import mnist
from torch.autograd import Variable

epochs = 5
batch_size = 200


# 以下函数是为了实现数据可视化的，可不必理解
def save_as_jpg(train_x, train_y, test_x, test_y):
    # 将MNIST数据集中的图片保存为jpg格式，将标签保存到对应的txt文档中

    flag=0
    path = "./MNIST_JPG/train/images"
    if not os.path.exists(path):
        os.makedirs(path)
        flag=1
    path = "./MNIST_JPG/train/labels"
    if not os.path.exists(path):
        os.makedirs(path)
        flag=1

    if flag==1:
        for i in range(train_x.shape[0]):
            cv.imwrite("./MNIST_JPG/train/images/{}.jpg".format(i), train_x[i, :, :])
            file_name = "./MNIST_JPG/train/labels/{}.txt".format(i)
            f = open(file_name, 'w')
            f.write(str(train_y[i]))
            f.close()


    flag=0
    path = "./MNIST_JPG/test/images"
    if not os.path.exists(path):
        os.makedirs(path)
        flag=1
    path = "./MNIST_JPG/test/labels"
    if not os.path.exists(path):
        os.makedirs(path)
        flag=1

    if flag==1:
        for i in range(test_x.shape[0]):
            cv.imwrite("./MNIST_JPG/test/images/{}.jpg".format(i), test_x[i, :, :])
            file_name = "./MNIST_JPG/test/labels/{}.txt".format(i)
            f = open(file_name, 'w')
            f.write(str(test_y[i]))
            f.close()
    print("\nSave as jpg has finished.")


class Net(nn.Module):
    '''
    Net是nn.Module的子类，继承了它的所有方法，包括__call__()方法和forward()方法，
    __call__()方法中会使用到前向转播函数forward()。
    '''

    def __init__(self):
        super(Net, self).__init__()  # 对继承自父类的属性进行初始化
        '''
        torch.nn和torch.nn.functional的不同在于nn只是定义了各种操作而并没有运行，
        而functional是运行各个操作。
        '''
        # 定义卷积操作
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # 定义随机失活操作
        # 定义全连接操作
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)
		# 下面的写法和上面的是等价的
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.layer3=nn.Sequential(
            nn.Linear(320, 60),
            nn.ReLU()
        )
        '''

    # 前向传播
    def forward(self, x):
        layer1 = F.relu(F.max_pool2d(self.conv1(x), 2))  # 卷积+最大池化+relu激活函数
        #layer1=F.relu(F.max_pool2d(F.conv2d(x,5),2))
        layer2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(layer1)), 2))
        layer2 = layer2.view(-1, 320)
        layer3 = F.relu(self.fc1(layer2))  # 全连接+relu激活函数
        # 可将以上四行代码换成封装好的以下四行代码，效果是相同的
        #layer1=self.layer1(x)
        #layer2 = self.layer2(x)
        #layer2 = x.view(-1, 320)
        #layer3 = self.layer3(x)
        layer4 = self.fc2(F.dropout(layer3, training=self.training))
        return F.log_softmax(layer4)  # 用softmax进行分类


# 模型的训练过程
def train0(model, epoch, train_x, train_y, num_samples, optimizer):
    num_batchs = int(num_samples / batch_size)
    model.train()  # 将模型设置为训练模式
    for batch in range(num_batchs):
        start, end = batch * batch_size, (batch + 1) * batch_size
        # 参数requires_grad=False的意思是不用求梯度
        data, target = Variable(train_x[start:end], requires_grad=False), Variable(train_y[start:end],
                                                                                   requires_grad=False)
        target = target.long()  # 转换为长整形
        optimizer.zero_grad()  # 将梯度初始化为0，否则每个batch的梯度会累积
        output = model(data)  # 会调用__call__()函数，进一步调用forward()函数进行前向传播
        print("target: ",target.shape)
        loss = F.nll_loss(output, target)  # 计算损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        if batch % 10 == 0:
            print("Train Epoch:{} [{}/{} ({:.2f}%)]\nLoss:{:.6f}".format(epoch, batch * len(data), num_samples,
                                                                         100.0 * batch_size * batch / num_samples,
                                                                         loss.data))
    torch.save(model, "mnist_torch.pkl")  # 保存训练好的模型


# 模型的测试过程
def test0(model):
    model.eval()  # 将模型设置为评价模式
    data, target = Variable(test_x), Variable(test_y)
    output = model(data)  # 前向传播，得到每个图片的预测分类
    target = target.long()
    test_loss = F.nll_loss(output, target).data  # 计算损失函数
    pred=torch.argmax(output.data,dim=1) # 求每个样本的最大值，即样本的预测类别
    correct = pred.eq(target.data).cpu().sum()  # 计算所有样本的正确率（正确分类个数/总个数），.cpu()表示用cpu进行计算
    # test_loss/=len(test_x) #平均损失
    print("\nTest set Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(test_x),
                                                                                 100.0 * correct / len(test_x)))

# template


def data_processing(train_x, train_y, test_x, test_y):

    # train_x是图像，共6w张28*28大小的图像；train_y是图像的真实分类
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    save_as_jpg(train_x, train_y, test_x, test_y)  # 把MNIST数据集保存为图片和文档


    # 将训练集和测试集转换为torch的张量
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)
    # 增加一个多余的维度 N, C, H, W
    train_x.resize_(train_x.shape[0], 1, 28, 28)
    test_x.resize_(test_x.shape[0], 1, 28, 28)
    return train_x, train_y, test_x, test_y


    

def train(model, optimizer, X, Y, epoch, batch_size):
    N = X.shape[0]
    steps = N // batch_size
    # shuffle X and Y
    idx = torch.randperm(N)
    X, Y = X[idx], Y[idx]
    for i in range(steps):
        
        start, end = i*batch_size, (i+1)*batch_size
        data, target = Variable(X[start:end], requires_grad=False), Variable(Y[start:end],
                                                                                   requires_grad=False)
        target = target.long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Train Epoch:{} [{}/{} ({:.2f}%)]\nLoss:{:.6f}".format(epoch, i * len(data), N,
                                                                         100.0 * batch_size * i / N,
                                                                         loss.data))



def test(model, X, Y):
    model.eval() ##
    data, target = Variable(X), Variable(Y)
    output = model(data)
    target = target.long()
    test_loss = F.nll_loss(output, target).data
    pred = torch.argmax(output.data, dim=1)

    correct = pred.eq(target.data).cpu().sum()
    accu = correct / len(X)
    # test_loss/=len(test_x) #平均损失
    print("\nTest set Average loss: {:.4f}, Accuracy: {} ({:.0f}%)\n".format(test_loss, accu,
                                                                                 100.0 * correct / len(test_x)))
    return accu

def data_shuffle(data):
    idx = torch.randperm(len(data))
    data = data[idx]
    return data

def data_split(data, label, prob):
    N = len(data)
    idx = torch.randperm(N)
    train_x, train_y = data[idx[:N*prob]], label[idx[:N*prob]]
    test_x, test_y = data[idx[N*prob:]], label[idx[N*prob:]]
    return train_x, train_y, test_x, test_y

def load_data(path, data_type):
    if data_type == "image":
        img2dataset(path)
    else:
        raise NotImplementedError

def img2dataset(img_folder, dtype="ndarray", ltype="one_hot", IMG_HEIGHT=20, IMG_WIDTH=20):
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv.imread( image_path, cv.COLOR_BGR2RGB)
            image=cv.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    if dtype == "tensor":
        img_data_array = torch.tensor(img_data_array)
    if ltype == "one_hot":
        pass
    elif ltype == "number":
        labels_dict = {label:idx for idx, label in enumerate(class_name)}
        labels_num = [labels_dict[label] for label in class_name]
        if dtype == "ndarray":
            labels = np.array(labels_num)
        elif dtype == "tensor":
            labels = torch.tensor(labels_num, dtype= long)
        return img_data_array, labels_num
    return img_data_array, labels, labels_dict
    

if __name__ == "__main__":

    
    path = ""
    data, label = load_data(path)
    train_x, train_y, test_x, test_y = data_split(data, label, 0.4)
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x, train_y, test_x, test_y = data_processing(train_x, train_y, test_x, test_y)
    N = train_x.shape[0]  # 数据个数
    # 1. initialize model
    model = Net()
    # 2. chooose optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # 3. train
    for epoch in range(1,epochs+1):
        train(model,epoch,train_x,train_y, N, optimizer) #训练
    model = torch.load("mnist_torch.pkl")  # 加载已经保存的模型
    # 4. test
    test(model, test_x, test_y)  # 在测试集上测试效果