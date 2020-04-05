import torch
import numpy as np
from torch import nn,optim
import torch.nn.functional as F
import torchvision.transforms as transforms
# https://github.com/ShusenTang/Dive-into-DL-PyTorch/tree/39328348c6a6dab4adc081b86aff814f0b35cb6d
# https://tangshusen.me/Dive-into-DL-PyTorch/#/
# https://github.com/dsgiitr/d2l-pytorch

# tensor
# pt = torch.tensor([[1,4],[2,1],[3,5]],dtype=torch.float)
# print (pt)
#tensor([[1, 4],[2, 1],[3, 5]])
# print(pt.shape) #torch.Size([3, 2])
# print (set(pt.shape)) #{2, 3}
# print (list(pt.shape)) #[3, 2]
# print (pt[0]) #tensor([1, 4])
# print (pt[0,1]) #tensor(4)
# print (pt.storage())
# 1
#  4
#  2
#  1
#  3
#  5
# [torch.LongStorage of size 6]
# pt_storage = pt.storage()
# pt_storage[0]=2
# print (pt)
# tensor([[2, 4],
#         [2, 1],
#         [3, 5]])
# second_pt = pt[1]

# print (second_pt) #tensor([2, 1])
# print (pt.storage_offset()) #0
# print (second_pt.storage_offset()) #2

# st_pt = pt[1]
# st_pt[0]=10
# print (pt)
# tensor([[ 1,  4],
#         [10,  1],
#         [ 3,  5]])

# st_pt_2 = pt[1].clone()
# st_pt_2[0]=10
# print (pt)
# tensor([[1, 4],
#         [2, 1],
#         [3, 5]])

# pt_t = pt.t()
# print (pt_t)
# tensor([[1, 2, 3],
#         [4, 1, 5]])

# print (id(pt.storage()) == id(pt_t.storage())) #True

# sm_tensor = torch.ones(3,4,5)
# sm_tensor_t = sm_tensor.transpose(0,2)
# print (sm_tensor.shape) #torch.Size([3, 4, 5])
# print (sm_tensor_t.shape) #torch.Size([5, 4, 3])
# print (sm_tensor.is_contiguous()) #True
# print (sm_tensor_t.is_contiguous()) #False
# print (pt.is_contiguous()) #True
# print(pt.t().is_contiguous()) #False

# Here’s a list of the possible values for the dtype argument:
#   torch.float32 or torch.float—32-bit floating-point
#   torch.float64 or torch.double—64-bit, double-precision floating-point   torch.float16 or torch.half—16-bit, half-precision floating-point
#   torch.int8—Signed 8-bit integers
#   torch.uint8—Unsigned 8-bit integers
#   torch.int16 or torch.short—Signed 16-bit integers
#   torch.int32 or torch.int—Signed 32-bit integers
#   torch.int64 or torch.long—Signed 64-bit integers

# Each of torch.float, torch.double, and so on has a corresponding concrete class 
# of torch.FloatTensor, torch.DoubleTensor, and so on. 
# The class for torch.int8 is torch.CharTensor, 
# and the class for torch.uint8 is torch.ByteTensor.
#  torch.Tensor is an alias for torch.FloatTensor. 
#  The default data type is 32-bit floating-point.
# double_points = torch.ones(10, 2, dtype=torch.double)
# short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)

# double_points = torch.zeros(10, 2).to(torch.double)
# short_points = torch.ones(10, 2).to(dtype=torch.short)

# pt_np = pt.numpy()
# print (pt_np)
# [[1 4]
#  [2 1]
#  [3 5]]

# pt_from_np_torch = torch.from_numpy(pt_np)
# print (pt_from_np_torch)
# tensor([[1, 4],
#         [2, 1],
#         [3, 5]])

# Serializing tensors
# torch.save(points, '../data/p1ch3/ourpoints.t')
# with open('../data/p1ch3/ourpoints.t','wb') as f:
#   torch.save(points, f)

# points = torch.load('../data/p1ch3/ourpoints.t')
# with open('../data/p1ch3/ourpoints.t','rb') as f:
#   points = torch.load(f)

# points_gpu = torch.tensor([[1,4],[2,1],[3,4]],device='cuda')
# pt_gpu = pt.to(device='cuda:0')
# At this point, any operation performed on the tensor, 
# such as multiplying all elements by a constant, is carried out on the GPU:
# pt_2 = 2*pt_gpu #multiplication performed on the GPU
# pt_cpu = pt_gpu.to(device='cpu')

# pt_gpu = pt.cuda()
# pt_gpu = pt.cuda(0)
# pt_cpu = pt_gpu.cpu()

# target = torch.tensor([1,3,2,5])
# target_onehot = torch.zeros(target.shape[0],10)
# target_onehot.scatter_(1,target.unsqueeze(1),1.0)
# print (target_onehot)
# # tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
# #         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
# #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
# #         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])

# print (pt)
# # tensor([[1, 4],[2, 1],[3, 5]])
# pt_unsqueeze = pt.unsqueeze(1)
# print (pt_unsqueeze)
# # tensor([[[1, 4]],[[2, 1]],[[3, 5]]])
# print (pt)
# print (torch.mean(pt,dim=0)) #tensor([2.0000, 3.3333])
# print (torch.mean(pt)) #tensor(2.6667)
# print (torch.mean(pt,dim=1)) #tensor([2.5000, 1.5000, 4.0000])
# print (torch.var(pt,dim=0)) #tensor([1.0000, 4.3333])

# less and equal
# bad_index = torch.le(pt,3)
# # print (bad_index)
# # tensor([[ True, False],
# #         [ True,  True],
# #         [ True, False]])
# bad_data = pt[bad_index]
# print (bad_data) #tensor([1., 2., 1., 3.])
# print (bad_index.shape,bad_index.dtype,bad_index.sum())
# # torch.Size([3, 2]) torch.bool tensor(4)

# le/ge/lt/gts


# x = torch.randn(4,2)
# y = x.view(-1,4)
# print (y.shape) #torch.Size([2, 4])
# z = x.view(8)
# print (z[0].item()) #0.07599391788244247 only one element tensors can be converted to Python scalars

# N, D_in, H, D_out = 64,1000,100,10
# x = np.random.randn(N,D_in)
# y = np.random.randn(N,D_out)
# w1 = np.random.randn(D_in,H)
# w2 = np.random.randn(H,D_out)
# learn_rate = 1e-6
# epoch = 500
# for i in range(epoch):
#   # Foorward passs
#   h = x.dot(w1)
#   h_relu = np.maximum(h,0)
#   y_pred = h_relu.dot(w2)
#   # compute loss
#   loss = np.square(y_pred-y).sum()
#   print (i,loss)
#   # backward pass
#   # compute the gradient
#   grad_y_pred = 2*(y_pred-y)
#   grad_w2 = h_relu.T.dot(grad_y_pred)
#   grad_h_relu = grad_y_pred.dot(w2.T)
#   grad_h = grad_h_relu
#   grad_h[h<0]=0
#   grad_w1 = x.T.dot(grad_h)
#   # update weights of w1 and w2
#   w1 -= learn_rate*grad_w1
#   w2 -= learn_rate*grad_w2

# N, D_in, H, D_out = 64,1000,100,10
# x = torch.randn(N,D_in)
# y = torch.randn(N,D_out)
# w1 = torch.randn(D_in,H)
# w2 = torch.randn(H,D_out)
# learn_rate = 1e-6
# epoch = 500
# for i in range(epoch):
#   # Foorward passs
#   h = x.mm(w1)
#   h_relu = h.clamp(min=0)
#   y_pred = h_relu.mm(w2)
#   # compute loss
#   loss =(y_pred-y).pow(2).sum().item()
#   print (i,loss)
#   # backward pass
#   # compute the gradient
#   grad_y_pred = 2*(y_pred-y)
#   grad_w2 = h_relu.t().mm(grad_y_pred)
#   grad_h_relu = grad_y_pred.mm(w2.t())
#   grad_h = grad_h_relu.clone()
#   grad_h[h<0]=0
#   grad_w1 = x.t().mm(grad_h)
#   # update weights of w1 and w2
#   w1 -= learn_rate*grad_w1
#   w2 -= learn_rate*grad_w2

# N, D_in, H, D_out = 64,1000,100,10
# x = torch.randn(N,D_in)
# y = torch.randn(N,D_out)
# w1 = torch.randn(D_in,H,requires_grad=True)
# w2 = torch.randn(H,D_out,requires_grad=True)
# learn_rate = 1e-6
# epoch = 500
# for i in range(epoch):
#   # Foorward passs
#   y_pred = x.mm(w1).clamp(min=0).mm(w2)
#   # compute loss
#   loss =(y_pred-y).pow(2).sum() #computation graph
#   print (i,loss.item())
#   loss.backward()
#   # backward pass
#   with torch.no_grad():
#     w1 -= learn_rate * w1.grad
#     w2 -= learn_rate * w2.grad
#     w1.grad.zero_()
#     w2.grad.zero_()

# import torch.nn as nn
# N, D_in, H, D_out = 64,1000,100,10
# x = torch.randn(N,D_in)
# y = torch.randn(N,D_out)
# model = torch.nn.Sequential(
#   torch.nn.Linear(D_in,H),
#   torch.nn.ReLU(),
#   torch.nn.Linear(H,D_out),
# )
# learn_rate = 1e-4
# epoch = 500
# loss_fn = torch.nn.MSELoss(reduction='sum')
# for i in range(epoch):
#   # Foorward passs
#   y_pred = model(x)
#   # compute loss
#   loss = loss_fn(y_pred,y) #computation graph
#   print (i,loss.item())
  
#   loss.backward()
#   # backward pass
#   with torch.no_grad():
#     for param in model.parameters():
#       param -= learn_rate*param.grad
#   model.zero_grad()

# import torch.nn as nn
# N, D_in, H, D_out = 64,1000,100,10
# x = torch.randn(N,D_in)
# y = torch.randn(N,D_out)
# model = torch.nn.Sequential(
#   torch.nn.Linear(D_in,H),
#   torch.nn.ReLU(),
#   torch.nn.Linear(H,D_out),
# )
# learn_rate = 1e-4
# epoch = 500
# loss_fn = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)
# for i in range(epoch):
#   # Foorward passs
#   y_pred = model(x)
#   # compute loss
#   loss = loss_fn(y_pred,y) #computation graph
#   print (i,loss.item())
#   optimizer.zero_grad()
#   loss.backward()
#   # backward pass
#   optimizer.step()
  
# import torch.nn as nn
# N, D_in, H, D_out = 64,1000,100,10
# x = torch.randn(N,D_in)
# y = torch.randn(N,D_out)
# class TwoLayerNet(torch.nn.Module):
#   def __init__(self, D_in, H, D_out):
#     super(TwoLayerNet,self).__init__()
#     self.linear1 = nn.Linear(D_in, H, bias = False)
#     self.linear2 = nn.Linear(H, D_out, bias = False)
#   def forward(self,x):
#     y_pred = self.linear2(self.linear1(x).clamp(min=0))
#     return y_pred

# model = TwoLayerNet(D_in, H, D_out)
# learn_rate = 1e-4
# epoch = 500
# loss_fn = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)
# for i in range(epoch):
#   # Foorward passs
#   y_pred = model(x)
#   # compute loss
#   loss = loss_fn(y_pred,y) #computation graph
#   print (i,loss.item())
#   optimizer.zero_grad()
#   loss.backward()
#   # backward pass
#   optimizer.step()

# import matplotlib.pyplot as plt
# import numpy as np

# x_data = [338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
# y_data = [640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]

# # init params
# b = -120
# w = -4
# lr = 1
# epoch = 100000 
# b_history=[b]
# w_history = [w]

# lr_b = 0
# lr_w = 0


# for i in range(epoch):
#   b_grad = 0.0
#   w_grad = 0.0
#   for j in range(len(x_data)):
#     b_grad = b_grad - 2.0*(y_data[j]-b-w*x_data[j])*(1.0)
#     w_grad = w_grad -  2.0*(y_data[j]-b-w*x_data[j])*(x_data[j])
#   # update params
#   lr_b = lr_b + b_grad**2
#   lr_w = lr_w + w_grad**2
#   b = b-lr/np.sqrt(lr_b)*b_grad
#   w = w-lr/np.sqrt(lr_w)*w_grad
#   b_history.append(b)
#   w_history.append(w)

# plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
# plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
# plt.xlim(-200,-100)
# plt.ylim(-5,5)
# plt.xlabel(r'$b$',fontsize=16)
# plt.ylabel(r'$w$',fontsize=16)
# plt.show()

"""
# demo1:词袋模型FizzBuzz
# FizzBuzz游戏规则如下：
# 从1开始往上数数，当遇到3的倍数的时候，说fizz，
# 当遇到5的倍数的时候说buzz，
# 当遇到15的倍数说fizzbuzz，其它情况下则正常数数。

def fizz_buzz_encode(i):
  if i % 15 == 0: return 3
  elif i % 5 == 0:return 2
  elif i % 3 == 0:return 1
  else: return 0

def fizz_buzz_decode(number,prediction):
  return [str(number),'fizz','buzz','fizzbuzz'][prediction]

def helper(i):
  print (fizz_buzz_decode(i,fizz_buzz_encode(i)))

# for i in range(20):
#   helper(i)

NUM_DIGITS = 10
def binary_encode(i,num_digits):
  return np.array([i >> d & 1 for d in range(num_digits)][::-1])

trX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(101,2**NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NUM_DIGITS)])
# print (trX.shape,trY.shape) #torch.Size([923, 10]) torch.Size([923])

NUM_HIDDEN = 100
model = torch.nn.Sequential(
  torch.nn.Linear(10,NUM_HIDDEN),
  torch.nn.ReLU(),
  torch.nn.Linear(NUM_HIDDEN,4)
)
if torch.cuda.is_available():
  model = model.cuda()
lossfn = torch.nn.CrossEntropyLoss()
optimer = torch.optim.SGD(model.parameters(),lr = 0.05)

BATCH_SIZE = 128

for epoch in range(10000):
  for start in range(0,len(trX),BATCH_SIZE):
    end = start+BATCH_SIZE
    batchX = trX[start:end]
    batchY = trY[start:end]
    if torch.cuda.is_available():
      batchX = batchX.cuda()
      batchY = batchY.cuda()
    y_pred = model(batchX)
    loss = lossfn(y_pred,batchY)
    optimer.zero_grad()
    print ('Epoch', epoch, loss.item())
    loss.backward()
    optimer.step()

testX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(1,100)])
with torch.no_grad():
  testY = model(testX)

predictions = zip(range(1,101), testY.max(1)[1].data.tolist())

print ([fizz_buzz_decode(i,x) for i,x in predictions])

"""

"""

"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from collections import Counter
# import numpy as np
# import random
# import math
# import pandas as pd
# import scipy
# import sklearn
# from sklearn.metrics.pairwise import cosine_distances
# USE_CUDA = torch.cuda.is_available()

# # 保持模型初始化的参数一致，这样每次训练的结果是一样的
# random.seed(1)
# np.random.seed(1)
# torch.manual_seed(1)
# if USE_CUDA:torch.cuda.manual_seed(1)


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import numpy as np

datasets里有两个函数需要定义：
1）__len__function需要返回整个数据集中有多少个item
2)__get__根据给定的index返回一个item


# class OwnDatasets(tud.Dataset):
#   def __init__(self,):
#     pass
#   def __len__(self,):
#     pass
#   def __getitem__(self,idx):
#     pass

# dataset = OwnDatasets()
# dataloader = tud.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle = True, num_workers=1)
"""


"""
# CNN image classification


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

mnist_data = datasets.MNIST('./mnist_data',train=True,download=True)

print (mnist_data[223][0].shape)

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1 = nn.Conv2d(1,20,5,1) #in_channels, out_channels, kernal_size, stride
    self.conv2 = nn.Conv2d(20,50,5,1)
    self.fc1 = nn.Linear(4*4*50, 500) #in_features, out_features, bias=True
    self.fc2 = nn.Linear(500,10)
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2) #kernal_size,stride=None,padding=0,dilation=1,return_indices=False(返回最大值的序号),ceil_mode=False
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2,2)
    x = x.view(-1,4*4*50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x,dim=1)

def train(model,device,train_loader,optimizer,epoch):
  # 把模型变为训练模式，因为它对batch_normalization和drop_out有影响
  model.train()
  for idx,(data,target) in enumerate(train_loader):
    data,target = data.to(device),target.to(device)
    pred = model(data)
    loss = F.nll_loss(pred,target) #cross_entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if idx % 100 == 0:
      print ("Train Epoch: {}, iter: {}, loss: {}".format(epoch,idx,loss.item()))

def test(model,device,test_loader):
  model.eval()
  total_loss = 0.
  correct = 0.
  with torch.no_grad():
    for idx,(data,target) in enumerate(test_loader):
      data,target = data.to(device),target.to(device)
      output = model(data)
      total_loss += F.nll_loss(output,target,reduction="sum").item()
      pred = output.argmax(dim=1)
      correct += pred.eq(target.view_as(pred)).sum().item()
  total_loss /= len(test_loader.dataset)
  acc = correct / len(test_loader.dataset)*100
  print("Test loss: {}, Accuracy: {}".format(total_loss,acc)) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
train_loader = torch.utils.data.DataLoader(
  datasets.MNIST("./mnist_data",train=True,download=True,transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))])),
  batch_size=batch_size,
  shuffle=True,
  num_workers=1,
  pin_memory=True)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST("./mnist_data",train=False,download=True,transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))])),
  batch_size=batch_size,
  shuffle=True,
  num_workers=1,
  pin_memory=True)

lr = 0.01
momentum = 0.5
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
num_epochs=100
for epoch in range(num_epochs):
  train(model,device,train_loader,optimizer,epoch)
  test(model,device,test_loader)

torch.save(model.state_dict(),'mnist_cnn.pt')

"""

"""
动手学深度学习-pytorch
中文版：https://github.com/ShusenTang/Dive-into-DL-Pytorch
英文版：https://github.com/dsgiitr/d2l-pytorch
视频：https://www.bilibili.com/video/av82045647?p=3
网页版：https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter02_prerequisite/2.1_install
"""

"""
2.数据操作
函数	功能
Tensor(*sizes)	基础构造函数
tensor(data,)	类似np.array的构造函数
ones(*sizes)	全1Tensor
zeros(*sizes)	全0Tensor
eye(*sizes)	对角线为1，其他为0
arange(s,e,step)	从s到e，步长为step
linspace(s,e,steps)	从s到e，均匀切分成steps份
rand/randn(*sizes)	均匀/标准分布
normal(mean,std)/uniform(from,to)	正态分布/均匀分布
randperm(m)	随机排列
"""

# 加法
# y = torch.rand(5,3)
# x = torch.eys(5,3)
# res = torch.empty(5,3)
# print (x+y)
# print (torch.add(x,y))
# print (torch.add(x,y,out=res))
# print (y.add_(x))

# 索引

# x = torch.eye(3,3)
# y = x[0,:]
# y += 1
# print (y) #tensor([2., 1., 1.])
# print (x[0:,])
# tensor([[2., 1., 1.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

# x = torch.randn(3,4)
# print (x)
# # tensor([[-0.2122, -0.3237, -0.7737,  1.2275],
# #         [-0.8009,  1.6314,  0.2569, -1.5025],
# #         [ 0.7174, -1.8894, -0.6839,  0.7775]])
# indices = torch.tensor([0,2])
# print(torch.index_select(x,0,indices))
# # tensor([[-0.2122, -0.3237, -0.7737,  1.2275],
# #         [ 0.7174, -1.8894, -0.6839,  0.7775]])
# print(torch.index_select(x,1,indices))
# # tensor([[-0.2122, -0.7737],
# #         [-0.8009,  0.2569],
# #         [ 0.7174, -0.6839]])

# x = torch.randn(3,4)
# print(x)
# # tensor([[ 0.0055, -0.9752, -1.0117,  0.1965],
# #         [-1.7644,  0.0634,  1.3276,  1.4400],
# #         [-0.5802,  0.4832, -0.1879,  2.6057]])
# mask = x.ge(0)
# print(mask)
# # Tensor([[ True, False, False,  True],
# #         [False,  True,  True,  True],
# #         [False,  True, False,  True]])
# print(torch.masked_select(x,mask))
# tensor([0.0055, 0.1965, 0.0634, 1.3276, 1.4400, 0.4832, 2.6057])

# x = torch.eye(3,4)
# print(x)
# # tensor([[1., 0., 0., 0.],
#         # [0., 1., 0., 0.],
#         # [0., 0., 1., 0.]])
# print(torch.nonzero(x))
# # tensor([[0, 0],
# #         [1, 1],
# #         [2, 2]])

# x = torch.tensor([[1,3,5],[2,4,6]])
# print (x)
# # tensor([[1, 3, 5],
# #         [2, 4, 6]])

# idx1 = torch.tensor([[0,1,0],[1,0,1]])
# res1 = torch.gather(x,0,idx1) #dim=0 表示列
# # print (res1)
# # tensor([[1, 4, 5],
# #         [2, 3, 6]])

# idx2 = torch.tensor([[0,1],[1,0]])
# res2 = torch.gather(x,1,idx2)
# # print(res2)
# # tensor([[1, 3],
# #         [4, 2]])

# 改变形状
# x = torch.randn(2,3)
# print(x)
# # tensor([[-0.8255,  1.4810,  0.2156],
# #         [-0.4836,  0.3341,  0.3162]])
# y = x.view(6)
# print(y)
# # tensor([-0.8255,  1.4810,  0.2156, -0.4836,  0.3341,  0.3162])
# z=x.view(-1,6)
# print(z)
# # tensor([[-0.8255,  1.4810,  0.2156, -0.4836,  0.3341,  0.3162]])
# d=x.view(3,2)
# print(d)
# # tensor([[-0.8255,  1.4810],
# #         [ 0.2156, -0.4836],
# #         [ 0.3341,  0.3162]])

# item()只适合一个数字的数组
# x = torch.randn(1)
# print(x)
# # tensor([1.6555])
# print(x.item())
# # 1.6555129289627075

"""
# 线性代数
函数	功能
trace	对角线元素之和(矩阵的迹)
diag	对角线元素
triu/tril	矩阵的上三角/下三角，可指定偏移量
mm/bmm	矩阵乘法，batch的矩阵乘法
addmm/addbmm/addmv/addr/baddbmm..	矩阵运算
t	转置
dot/cross	内积/外积
inverse	求逆矩阵
svd	奇异值分解
"""

# # 广播机制
# x = torch.arange(1,3).view(1,2)
# # print(x) #tensor([[1, 2]])
# y = torch.arange(1,4).view(3,1)
# print(y)
# # tensor([[1],
# #         [2],
# #         [3]])
# print (x+y)
# # tensor([[2, 3],
# #         [3, 4],
# #         [4, 5]])

# tensor转numpy
# a = torch.ones(2,3)
# b = a.numpy()
# # print(b)
# # [[1. 1. 1.]
# #  [1. 1. 1.]]
# c = torch.from_numpy(b)
# print(c)
# # tensor([[1., 1., 1.],
# #         [1., 1., 1.]])

"""
上一节介绍的Tensor是这个包的核心类，如果将其属性.requires_grad设置为True，它将开始追踪(track)在其上的所有操作
（这样就可以利用链式法则进行梯度传播了）。完成计算后，可以调用.backward()来完成所有梯度计算。此Tensor的梯度将累积到.grad属性中。
注意在y.backward()时，如果y是标量，则不需要为backward()传入任何参数；否则，需要传入一个与y同形的Tensor。解释见 2.3.2 节。

如果不想要被继续追踪，可以调用.detach()将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了。
此外，还可以用with torch.no_grad()将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，
我们并不需要计算可训练参数（requires_grad=True）的梯度。

"""


# x = torch.ones(2,2,requires_grad=True)
# y = x+2
# z = y * y *3
# out = z.mean()
# out.backward()
# print(x.grad)
# # tensor([[4.5000, 4.5000],
# #         [4.5000, 4.5000]])

# # 再来反向传播一次，注意grad是累加的
# out2 = x.sum()
# out2.backward()
# print(x.grad)
# # tensor([[5.5000, 5.5000],
# #         [5.5000, 5.5000]])

# out3 = x.sum()
# x.grad.data.zero_()
# out3.backward()
# # print(x.grad)
# # tensor([[1., 1.],
# #         [1., 1.]])

# a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
# a = ((a * 3) / (a - 1))
# print(a.requires_grad) # False
# a.requires_grad_(True)
# print(a.requires_grad) # True
# b = (a * a).sum()
# print(b.grad_fn)

"""
# 3.2 线性回归的从零开始实现

import numpy as np
import random
from matplotlib import pyplot as plt

num_inputes = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.randn(num_examples,num_inputes,dtype=torch.float32)
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)
batch_size = 10


# plt.scatter(features[:,0].numpy(),labels.numpy(),1)
# plt.show()

def data_iter(batch_size, features, labels):
  num_examples = len(features)
  indices = list(range(num_examples))
  random.shuffle(indices)
  for i in range(0,num_examples,batch_size):
    j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
    yield features.index_select(0,j),labels.index_select(0,j)

# for X,y in data_iter(batch_size,features,labels):
#   print (X,y)
#   break
  
w = torch.tensor(np.random.normal(0,0.01,(num_inputes,1)),dtype=torch.float32,requires_grad=True)
b = torch.zeros(1,dtype=torch.float32,requires_grad=True)

def linreg(X,w,b):
  return torch.mm(X,w)+b

def squared_loss(y_hat,y):
  return (y_hat-y.view(y_hat.size()))**2 / 2

def sgd(params, lr, batch_size):
  for param in params:
    param.data -= lr*param.grad/batch_size

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
  for X,y in data_iter(batch_size,features,labels):
    l = loss(net(X,w,b),y).sum()
    l.backward()
    sgd([w,b],lr,batch_size)
    w.grad.data.zero_()
    b.grad.data.zero_()
  train_l = loss(net(features, w, b), labels)
  print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

# epoch 1, loss 0.035633
# epoch 2, loss 0.000127
# epoch 3, loss 0.000049
"""

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True)) #tensor([[5, 7, 9]])
# print(X.sum(dim=1, keepdim=True))
# # tensor([[ 6],
# #         [15]])


# def softmax(X):
#   x_exp = X.exp()
#   partition = x_exp.sum(dim=1,keepdim=True)
#   return x_exp / partition #广播机制

# x = torch.rand((2,5))
# x_prob = softmax(x)
# print (x_prob, x_prob.sum(dim=1))
# # tensor([[0.1560, 0.2762, 0.1418, 0.2789, 0.1471],
# #         [0.2560, 0.1560, 0.2322, 0.1148, 0.2410]])
# # tensor([1.0000, 1.0000])


"""
# 权重衰减

import torch.nn as nn
# import sys
# sys.path.append("..") 
# import d2lzh_pytorch as d2l
# from matplotlib import pyplot as plt


n_train, n_test, num_inputs = 20, 100, 200
batch_size, num_epochs, lr = 1, 100, 0.003
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def init_params():
  w = torch.randn((num_inputs,1),requires_grad=True)
  b = torch.zeros(1,requires_grad=True)
  return [w,b]

def l2_penalty(w):
  return (w**2).sum() / 2


def linreg(x,w,b):
  return torch.mm(x,w)+b

def squared_loss(y_hat,y):
  return (y_hat-y.view(y_hat.size()))**2/2

def sgd(params, lr, batch_size):
  for p in params:
    p.data -= lr*p.grad.data/batch_size

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5,2.5)):
  plt.rcParams['figure.figsize'] = figsize
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.semilogy(x_vals, y_vals)
  if x2_vals and y2_vals:
    plt.semilogy(x2_vals,y2_vals,linestyle=':')
    plt.legend(legend)



net, loss = linreg,squared_loss

def fit_and_plot(lambd):
  w,b = init_params()
  train_ls, test_ls = [],[]
  for _ in range(num_epochs):
    for x,y in train_iter:
      l = loss(net(x,w,b),y)+lambd*l2_penalty(w)
      l = l.sum()
      if w.grad is not None:
        w.grad.data.zero_()
        b.grad.data.zero_()
      l.backward()
      sgd([w,b],lr,batch_size)
    train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
    test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
  semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
  print('L2 norm of w:', w.norm().item())

def fit_and_plot_pytorch(wd):
  # 对权重参数衰减。权重名称一般是以weight结尾
  net = nn.Linear(num_inputs,1)
  nn.init.normal_(net.weight,mean=0,std=1)
  nn.init.normal_(net.bias,mean=0,std=1)
  optimizer_w = torch.optim.SGD(params=[net.weight],lr=lr,weight_decay=wd)
  optimizer_b = torch.optim.SGD(params=[net.bias],lr=lr)
  train_ls, test_ls = [],[]
  for _ in range(num_epochs):
    for x,y in train_iter:
      l = loss(net(x),y).mean()
      optimizer_w.zero_grad()
      optimizer_b.zero_grad()
      l.backward()
      optimizer_w.step()
      optimizer_b.step()
    train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
    test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
  semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
  print('L2 norm of w:', w.norm().item())

""" 

"""
# dropout


def dropout(x,drop_prob):
  x = x.float()
  assert 0 <= drop_prob <= 1
  keep_prob = 1-drop_prob
  if keep_prob == 0: return torch.zeros_like(x)
  mask = (torch.rand(x.shape)<keep_prob).float()
  return mask * x / keep_prob

x = torch.randn((2,3))
print(x)
# tensor([[-0.3235,  2.6317,  0.6783],
#         [ 0.9670,  0.3266, -1.0439]]) 
print(dropout(x,0.5))
# tensor([[-0.6471,  5.2634,  1.3567],
#         [ 0.0000,  0.0000, -2.0878]]) 

print(dropout(x,0.7))
# tensor([[-1.0784,  0.0000,  2.2611],
#         [ 0.0000,  0.0000, -0.0000]])

net = nn.Sequential(
  d2l.FlattenLayer(),
  nn.Linear(num_inputs, num_hiddens1),
  nn.ReLU(),
  nn.Dropout(drop_prob1),
  nn.Linear(num_hiddens1, num_hiddens2), 
  nn.ReLU(),
  nn.Dropout(drop_prob2),
  nn.Linear(num_hiddens2, 10))

for param in net.parameters():nn.init.normal_(param, mean=0, std=0.01)
"""


"""
深度网络模型
"""
# import torch.nn as nn
# Module类是nn模块里提供的一个模型构造类，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型。
# class MLP(nn.Module):
#   def __init__(self, **kwargs):
#     super(MLP, self).__init__(**kwargs)
#     self.hidden = nn.Linear(784,256)
#     self.act = nn.ReLU()
#     self.output = nn.Linear(256,10)
#   # 定义模型的前向计算
#   def forward(self,x):
#     a = self.act(self.hidden(x))
#     return self.output(a)

# net = MLP()
# print (net)
# MLP(
#   (hidden): Linear(in_features=784, out_features=256, bias=True)
#   (act): ReLU()
#   (output): Linear(in_features=256, out_features=10, bias=True)
# )

# x = torch.rand(2,784)
# print(net(x))

"""
Module类是一个通用的部件。事实上，PyTorch还实现了继承自Module的可以方便构建模型的类: 
如Sequential、ModuleList和ModuleDict等等。
1)当模型的前向计算为简单串联各个层的计算时，Sequential类可以通过更加简单的方式定义模型。
这正是Sequential类的目的：它可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数来逐一添加Module的实例，
而模型的前向计算就是将这些实例按添加的顺序逐一计算。
2）ModuleList接收一个子模块的列表作为输入，然后也可以类似List那样进行append和extend操作.
3)既然Sequential和ModuleList都可以进行列表化构造网络，那二者区别是什么呢。
ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），
而且没有实现forward功能需要自己实现，所以上面执行net(torch.zeros(1, 784))会报NotImplementedError；
而Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现。
4)ModuleList不同于一般的Python的list，加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中。
"""

# net = nn.Sequential(
#   nn.Linear(784, 256),
#   nn.ReLU(),
#   nn.Linear(256, 10), 
#   )
# print(net)
# net(X)

# net = nn.ModuleList([nn.Linear(784,256),nn.ReLU()])
# net.append(nn.Linear(256,10))
# print (net[-1])
# # Linear(in_features=256, out_features=10, bias=True)
# print (net)
# # ModuleList(
# #   (0): Linear(in_features=784, out_features=256, bias=True)
# #   (1): ReLU()
# #   (2): Linear(in_features=256, out_features=10, bias=True)
# # )
# net(torch.zeros(1, 784)) # 会报NotImplementedError

# class Module_ModuleList(nn.Module):
#   def __init__(self):
#     super(Module_ModuleList, self).__init__()
#     self.linears = nn.ModuleList([nn.Linear(10, 10)])

# class Module_List(nn.Module):
#   def __init__(self):
#     super(Module_List, self).__init__()
#     self.linears = [nn.Linear(10, 10)]

# # net1 = Module_ModuleList()
# # net2 = Module_List()

# # print("net1:")
# # for p in net1.parameters():
# #     print(p.size())
# # # net1:
# # # torch.Size([10, 10])
# # # torch.Size([10])

# # print("net2:")
# # for p in net2.parameters():
# #     print(p)
# # # net2:


# net = nn.ModuleDict({
#     'linear': nn.Linear(784, 256),
#     'act': nn.ReLU(),
# })
# net['output'] = nn.Linear(256, 10) # 添加
# print(net['linear']) # 访问
# print(net.output)
# print(net)
# # net(torch.zeros(1, 784)) # 会报NotImplementedError

# # Linear(in_features=784, out_features=256, bias=True)
# # Linear(in_features=256, out_features=10, bias=True)
# # ModuleDict(
# #   (act): ReLU()
# #   (linear): Linear(in_features=784, out_features=256, bias=True)
# #   (output): Linear(in_features=256, out_features=10, bias=True)
# # )

# net = nn.Sequential(
#   nn.Linear(4, 3), 
#   nn.ReLU(), 
#   nn.Linear(3, 1))  # pytorch已进行默认初始化
# for name, param in net.named_parameters():
#   print (name,param.size())
# # 0.weight torch.Size([3, 4])
# # 0.bias torch.Size([3])
# # 2.weight torch.Size([1, 3])
# # 2.bias torch.Size([1])

# class MyModel(nn.Module):
#     def __init__(self, **kwargs):
#         super(MyModel, self).__init__(**kwargs)
#         self.weight1 = nn.Parameter(torch.rand(20, 20))
#         self.weight2 = torch.rand(20, 20)
#     def forward(self, x):
#         pass

# n = MyModel()
# for name, param in n.named_parameters():
#     print(name) #weight1

"""
不含模型参数的自定义层
"""
# from torch import nn

# class CenterLayer(nn.Module):
#   def __init__(self, **kwargs):
#     super(CenterLayer, self).__init__(**kwargs)
#   def forward(self, x):
#     return x-x.mean()

# layer = CenterLayer()
# # print(layer(torch.tensor([1,2,3,4,5],dtype=torch.float))) #tensor([-2., -1.,  0.,  1.,  2.])

# net = nn.Sequential(nn.Linear(8,128),CenterLayer())
# y = net(torch.rand(4,8))
# print (y.mean().item()) #-1.0244548320770264e-08


"""
如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里。所以在自定义含模型参数的层时，我们应该将参数定义成Parameter，
可以使用ParameterList和ParameterDict分别定义参数的列表和字典。
1)访问模型参数：
for name,param in net.named_parameters():
  print (name,param.size())

2)访问网络某一层的参数
for name,param in net[1].named_parameters():
  print (name,param.size())

3)新建网络参数--使用nn.Parameter
class MyModule(nn.Module):
  def __init__(self, **kwargs):
    super(MyModule, self).__init__(**kwargs)
    self.weight1 = nn.Parameter(torch.rand(20,20))
    self.weight2 = torch.rand(20,20)
  def forward(self,x):
    pass

n = MyModule()
for name,param in n.named_parameters():
  print(name) #weight1

4）基于nn.ParameterList
class MyDense(nn.Module):
  def __init__(self):
    super(MyDense, self).__init__()
    self.params = nn.ParameterList([nn.Parameter(torch.randn(4,4)) for i in range(3)])
    self.params.append(nn.Parameter(torch.randn(4,1)))
  def forward(self, x):
    for i in range(len(self.params)):
      x = torch.mm(x,self.params[i])
    return x

# net = MyDense()
# print (net)
# MyDense(
#   (params): ParameterList(
#       (0): Parameter containing: [torch.FloatTensor of size 4x4]
#       (1): Parameter containing: [torch.FloatTensor of size 4x4]
#       (2): Parameter containing: [torch.FloatTensor of size 4x4]
#       (3): Parameter containing: [torch.FloatTensor of size 4x1]
#   )
# )

5）ParameterDict()
ParameterDict接收一个Parameter实例的字典作为输入然后得到一个参数字典，然后可以按照字典的规则使用。
例如update()新增参数，使用keys()返回所有键值，使用items()返回所有键值对等。

class MyDictDense(nn.Module):
  def __init__(self):
    super(MyDictDense, self).__init__()
    self.params = nn.ParameterDict({
      'linear1':nn.Parameter(torch.randn(4,4)),
      'linear2':nn.Parameter(torch.randn(4,1))
    })
    self.params.update({'linear3':nn.Parameter(torch.randn(4,2))})
  def forward(self,x,choice='linear1'):
    return torch.mm(x, self.params[choice])

# net = MyDictDense()
# print(net)
# MyDictDense(
#   (params): ParameterDict(
#       (linear1): Parameter containing: [torch.FloatTensor of size 4x4]
#       (linear2): Parameter containing: [torch.FloatTensor of size 4x1]
#       (linear3): Parameter containing: [torch.FloatTensor of size 4x2]
#   )
# )

"""

"""
变量存取
# x = torch.ones(3)
# torch.save(x,'x.pt')
# x2 = torch.load('x.pt')
# print (x2)

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
# xy_list
# [tensor([1., 1., 1.]), tensor([0., 0., 0., 0.])]
"""

"""
--读写模型
在PyTorch中，Module的可学习参数(即权重和偏差)，模块模型包含在参数中(通过model.parameters()访问)。
state_dict是一个从参数名称隐射到参数Tesnor的字典对象。
# class MLP(nn.Module):
#   def __init__(self):
#     super(MLP,self).__init__()
#     self.hidden = nn.Linear(3,2)
#     self.act = nn.ReLU()
#     self.output = nn.Linear(2,1)
#   def forward(self,x):
#     a = self.act(self.hidden(x))
#     return self.output(a)

# # net = MLP()
# # print (net.state_dict())
# # OrderedDict([('hidden.weight', tensor([[ 0.2448,  0.1856, -0.5678],
# #                       [ 0.2030, -0.2073, -0.0104]])),
# #              ('hidden.bias', tensor([-0.3117, -0.4232])),
# #              ('output.weight', tensor([[-0.4556,  0.4084]])),
# #              ('output.bias', tensor([-0.3573]))])

注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。优化器(optim)也有一个state_dict，
其中包含关于优化器状态以及所使用的超参数的信息。

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.state_dict()
输出：

{'param_groups': [{'dampening': 0,
   'lr': 0.001,
   'momentum': 0.9,
   'nesterov': False,
   'params': [4736167728, 4736166648, 4736167368, 4736165352],
   'weight_decay': 0}],
 'state': {}}

"""


"""
PyTorch中保存和加载训练模型有两种常见的方法:

仅保存和加载模型参数(state_dict)；
保存和加载整个模型。
1. 保存和加载state_dict(推荐方式)
保存：
torch.save(model.state_dict(), PATH) # 推荐的文件后缀名是pt或pth
Copy to clipboardErrorCopied
加载：
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
2. 保存和加载整个模型
保存：

torch.save(model, PATH)
Copy to clipboardErrorCopied
加载：

model = torch.load(PATH)
X = torch.randn(2, 3)
Y = net(X)

PATH = "./net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
Y2 == Y

"""

"""
# 卷积运算


def corr2d(x,k):
  h,w = k.shape
  y = torch.zeros((x.shape[0]-h+1, x.shape[1]-w+1))
  for i in range(y.shape[0]):
    for j in range(y.shape[1]):
      y[i,j] = ((x[i:i+h,j:j+w])*k).sum()
  return y

class Conv2D(nn.Module):
  def __init__(self, kernal_size):
    super(Conv2D, self).__init__()
    self.weight = nn.Parameter(torch.randn(kernal_size))
    self.bias = nn.Parameter(torch.randn(1))
  def forward(self, x):
    return corr2d(x,self.weight)+self.bias

conv2d = Conv2D(kernal_size=(1,2))
step = 40
lr = 0.01
x = torch.ones(6, 8)
x[:,2:6] = 0
K = torch.tensor([[1, -1]])
y = corr2d(x,K)
for i in range(step):
  y_hat = conv2d(x)
  l = ((y_hat-y)**2).sum()
  l.backward()

  conv2d.weight.data -= lr*conv2d.weight.grad
  conv2d.bias.data -= lr*conv2d.bias.grad

  conv2d.weight.grad.fill_(0)
  conv2d.bias.grad.fill_(0)
  if (i + 1) % 5 == 0:
    print('Step %d, loss %.3f' % (i + 1, l.item()))


# Step 5, loss 10.664
# Step 10, loss 2.807
# Step 15, loss 0.764
# Step 20, loss 0.211
# Step 25, loss 0.058
# Step 30, loss 0.016
# Step 35, loss 0.005
# Step 40, loss 0.001

"""

"""
# 多输入通道

def corr2d(x,k):
  h,w = k.shape
  y = torch.zeros((x.shape[0]-h+1, x.shape[1]-w+1))
  for i in range(y.shape[0]):
    for j in range(y.shape[1]):
      y[i,j] = ((x[i:i+h,j:j+w])*k).sum()
  return y

def corr2d_multi_in(x,k):
  x.shape:D*H*W
  res = corr2d(x[0,:,:],k[0,:,:])
  for i in range(1,x.shape[0]):
    res += corr2d(x[i,:,:],k[i,:,:])
  return res

# 多输出通道

def corr2d_multi_in_out(x,K):
  # filter_shape:cout*cin*D*H*w
  return torch.stack([corr2d_multi_in(x,k) for k in range(K)])


def corr2d_multi_in_out_1x1(X,K):
  c_in,h,w=X.shape
  c_out = K.shape[0]
  X = X.view(c_in, h*w)
  K = K.view(c_out, c_in)
  Y = torch.mm(K,X) #全连接层的矩阵乘法
  return Y.view(c_out,h,w)
"""
"""
# 池化层

def pool2d(x, pool_size, mode='max'):
  x = x.float()
  p_h, p_w = pool_size
  y = torch.zeros(x.shape[0]-p_h+1, x.shape[1]-p_w+1)
  for i in range(y.shape[0]):
    for j in range(y.shape[1]):
      if mode == 'max':
        y[i,j] = x[i:i+p_h,j:j+p_w].max()
      elif mode == 'avg':
        y[i,j] = x[i:i+p_h,j:j+p_w].mean()
  return y

# x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# print (pool2d(x,(2,2),'max'))
# # tensor([[4., 5.],
# #         [7., 8.]])
# print (pool2d(x,(2,2),'avg'))
# # tensor([[2., 3.],
# #         [5., 6.]])

# 池化层填充和步幅
# x = torch.arange(16,dtype=torch.float32).view((1,1,4,4))
# print (x)
# # tensor([[[[ 0.,  1.,  2.,  3.],
# #           [ 4.,  5.,  6.,  7.],
# #           [ 8.,  9., 10., 11.],
# #           [12., 13., 14., 15.]]]])

# # 无填充，默认步幅为2
# pool2d_res = nn.MaxPool2d(3)
# print(pool2d_res(x))
# # tensor([[[[10.]]]])

# # 有填充
# pool2d_res_padding = nn.MaxPool2d(3,padding=1,stride=2)
# print(pool2d_res_padding(x))
# # tensor([[[[ 5.,  7.],
# #           [13., 15.]]]])

# 当然，我们也可以指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅。

# pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
# pool2d(X)
# # tensor([[[[ 1.,  3.],
# #           [ 9., 11.],
# #           [13., 15.]]]])
"""


"""
# LeNet模型


# import sys
# sys.path.append("..") 
# import d2lzh_pytorch as d2l
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1,6,5),
      nn.Sigmoid(),
      nn.MaxPool2d(2,2),
      nn.Conv2d(6,16,5),
      nn.Sigmoid(),
      nn.MaxPool2d(2,2)
    )
    self.fc = nn.Sequential(
      nn.Linear(16*5*5,120),
      nn.Sigmoid(),
      nn.Linear(120,84),
      nn.Sigmoid(),
      nn.Linear(84,10)
    )
  def forward(self,img):
    feature = self.conv(img)
    output = self.fc(feature.view(img.shape[0], -1))
    return output


net = LeNet()
batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy(data_iter, net, device=None):
  if device is None and isinstance(net,torch.nn.Module):
    device = list(net.parameters())[0].device
  acc_sum, n = 0.0, 0
  with torch.no_grad():
    for x,y in data_iter:
      if isinstance(net, torch.nn.Module):
        # 评估模式
        net.eval()
        acc_sum += (net((x.to(device))).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
        #训练模式
        net.train()
      else:
        if('is_training' in net.__code__.co_varnames): 
          # 如果有is_training这个参数
          # # 将is_training设置成False
          acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
        else:
          acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
      n += y.shape[0]
    return acc_sum / n

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
  net = net.to(device)
  loss = nn.CrossEntropyLoss()
  for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
    for x,y in train_iter:
      x = x.to(device)
      y = y.to(device)
      y_hat = net(x)
      l = loss(y_hat, y)
      optimizer.zero_grad()
      l.backward()
      optimizer.step()
      train_l_sum += l.cpu().item()
      n += y.shape[0]
      batch_count += 1
    test_acc = evaluate_accuracy(test_iter,net)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

"""

"""
# VGG
# import sys
# sys.path.append("..") 
# import d2lzh_pytorch as d2l
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlattenLayer(nn.Module):
  def __init__(self):
    super(FlattenLayer, self).__init__()
  def forward(self, x):
    return x.view(x.shape[0],-1)


def vgg_block(num_convs, in_channels, out_channels):
  blk = []
  for i in range(num_convs):
    if i == 0:
      blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
    else:
      blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
    blk.append(nn.ReLU())
  blk.append(nn.MaxPool2d(kernel_size=2,stride=2))
  return nn.Sequential(*blk)
# VGG16
conv_arch = ((2, 1, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

def vgg16(conv_arch, fc_features, fc_hidden_units=4096):
  net = nn.Sequential()
  for i ,(num_convs,in_channels,out_channels) in enumerate(conv_arch):
    net.add_module('vgg16_block_'+str(i+1), vgg_block(num_convs,in_channels,out_channels))
  net.add_module('fc',nn.Sequential(
    FlattenLayer(),
    nn.Linear(fc_features,fc_hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(fc_hidden_units,fc_hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(fc_hidden_units,10)
  ))
  return net

net = vgg16(conv_arch,fc_features,fc_hidden_units)
X = torch.rand(1,1,224,224)

# named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
# for name, blk in net.named_children():
#   X = blk(X)
#   print(name, 'output shape: ', X.shape)

# # vgg16_block_1 output shape:  torch.Size([1, 64, 112, 112])
# # vgg16_block_2 output shape:  torch.Size([1, 128, 56, 56])
# # vgg16_block_3 output shape:  torch.Size([1, 256, 28, 28])
# # vgg16_block_4 output shape:  torch.Size([1, 512, 14, 14])
# # vgg16_block_5 output shape:  torch.Size([1, 512, 7, 7])
# # fc output shape:  torch.Size([1, 10])


# batch_size = 64
# # 如出现“out of memory”的报错信息，可减小batch_size或resize
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# lr, num_epochs = 0.001, 5
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

"""




"""
# 网络中的网络（NiN）

# import sys
# sys.path.append("..") 
# import d2lzh_pytorch as d2l
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nin_block(in_channels, out_channels, kernal_size, stride, padding):
  blk = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding),
    nn.ReLU(),
    nn.Conv2d(out_channels,out_channels,kernal_size=1),
    nn.ReLU(),
    nn.Conv2d(out_channels,out_channels,kernal_size=1),
    nn.ReLU()
  )
  return blk
"""



"""
# 含并行连接的网络（GoogleNet）


# import sys
# sys.path.append("..") 
# import d2lzh_pytorch as d2l
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Inception(nn.Module):
  # c1-c4为每条线路里的层的输出通道数
  def __init__(self, in_c, c1, c2, c3, c4):
    super(Inception, self).__init__()
    # 线路1，单1 x 1卷积层
    self.p1_1 = nn.Conv2d(in_c,c1,kernel_size=1)
    # 线路2，1 x 1卷积层后接3 x 3卷积层
    self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
    self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
    # 线路3，1 x 1卷积层后接5 x 5卷积层
    self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
    self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5, padding=2)
    # 线路4，3 x 3最大池化层后接1 x 1卷积层
    self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
    self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
  def forward(self,x):
    p1 = F.relu(self.p1_1(x))
    p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
    p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
    p4 = F.relu(self.p4_2(self.p4_1(x)))
    # 在通道维上连接输出
    return torch.cat((p1,p2,p3,p4), dim=1)

b1 = nn.Sequential(
  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
  nn.ReLU(),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  )

b2 = nn.Sequential(
  nn.Conv2d(64, 64, kernel_size=1),
  nn.Conv2d(64, 192, kernel_size=3, padding=1),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  )

# 第三模块串联2个完整的Inception块。
# 第一个Inception块的输出通道数为64+128+32+32=256,其中4条线路的输出通道数比例为64:128:32:32=2:4:1:1
# 其中第二、第三条线路先分别将输入通道数减小至96/192=1/2和16/192=1/1216/192=1/12后，再接上第二层卷积层。
# 第二个Inception块输出通道数增至128+192+96+64=480，每条线路的输出通道数之比为128:192:96:64=4:6:3:2
# 其中第二、第三条线路先分别将输入通道数减小至128/256=1/2和32/256=1/8。
b3 = nn.Sequential(
  Inception(192, 64, (96, 128), (16, 32), 32),
  Inception(256, 128, (128, 192), (32, 96), 64),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  )

# 第四模块更加复杂。它串联了5个Inception块，
# 其输出通道数分别是192+208+48+64=512、160+224+64+64=512、128+256+64+64=512、112+288+64+64=528和256+320+128+128=832。
# 这些线路的通道数分配和第三模块中的类似，首先含3×3卷积层的第二条线路输出最多通道，其次是仅含1×1卷积层的第一条线路，
# 之后是含5×5卷积层的第三条线路和含3×3最大池化层的第四条线路。
# 其中第二、第三条线路都会先按比例减小通道数。这些比例在各个Inception块中都略有不同。

b4 = nn.Sequential(
  Inception(480, 192, (96, 208), (16, 48), 64),
  Inception(512, 160, (112, 224), (24, 64), 64),
  Inception(512, 128, (128, 256), (24, 64), 64),
  Inception(512, 112, (144, 288), (32, 64), 64),
  Inception(528, 256, (160, 320), (32, 128), 128),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第五模块有输出通道数为256+320+128+128=832和384+384+128+128=1024的两个Inception块。
# 其中每条线路的通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同。
# 需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均池化层来将每个通道的高和宽变成1。
# 最后我们将输出变成二维数组后接上一个输出个数为标签类别数的全连接层。
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
class FlattenLayer(nn.Module):
  def __init__(self):
    super(FlattenLayer, self).__init__()
  def forward(self, x):
    return x.view(x.shape[0],-1)
b5 = nn.Sequential(
  Inception(832, 256, (160, 320), (32, 128), 128),
  Inception(832, 384, (192, 384), (48, 128), 128),
  GlobalAvgPool2d())

# net = nn.Sequential(b1, b2, b3, b4, b5, FlattenLayer(), nn.Linear(1024, 10))
# X = torch.rand(1,1,224,224)
# for blk in net.children():
#   X = blk(X)
#   print('output shape: ', X.shape)
# # output shape:  torch.Size([1, 64, 56, 56])
# # output shape:  torch.Size([1, 192, 28, 28])
# # output shape:  torch.Size([1, 480, 14, 14])
# # output shape:  torch.Size([1, 832, 7, 7])
# # output shape:  torch.Size([1, 1024, 1, 1])
# # output shape:  torch.Size([1, 1024])
# # output shape:  torch.Size([1, 10])
"""

"""
# 批量归一化-batchnormalization
def batch_norm(is_training, x, gamma, beta, moving_mean, moving_var, eps, momentum):
  if not is_training:
    # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
    x_hat = (x-moving_mean) / torch.sqrt(moving_var+eps)
  else:
    assert len(X.shape) in (2, 4)
    if len(x.shape) == 2:
      # 使用全连接层的情况，计算特征维上的均值和方差
      mean = x.mean(dim=0)
      var = ((x - mean) ** 2).mean(dim=0)
    else:
      # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持;
      # X的形状以便后面可以做广播运算
      mean = x.mean(dim=0, keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
      var = ((x-mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
    x_hat = (x-mean) / torch.sqrt(var+eps)
    # 更新移动平均的均值和方差
    moving_mean = momentum * moving_mean + (1.0-momentum)*mean
    moving_var = momentum * moving_var + (1.0-momentum)*var
  y = gamma * x_hat + beta
  return y, moving_mean, moving_var

class BatchNorm(nn.Module):
  def __init__(self, num_features, num_dims):
    super(BatchNorm, self).__init__()
    if num_dims == 2:
      shape = (1,num_features)
    else:
      shape = (1,num_features,1,1)
    # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
    self.gamma = nn.Parameter(torch.ones(shape))
    self.beta = nn.Parameter(torch.zeros(shape))
    self.moving_mean = torch.zeros(shape)
    self.moving_var = torch.zeros(shape)
  def forward(self, x):
    # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
    if self.moving_mean.device != X.device:
      self.moving_mean = self.moving_mean.to(X.device)
      self.moving_var = self.moving_var.to(X.device)
      # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
      Y, self.moving_mean, self.moving_var = batch_norm(self.training, x, self.gamma, self.beta, 
      self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
    return Y

# pytorch中nn模块定义的BatchNorm1d和BatchNorm2d类使用起来更加简单，
# 二者分别用于全连接层和卷积层，都需要指定输入的num_features参数值。
# net = nn.Sequential(
#             nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
#             nn.BatchNorm2d(6),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2), # kernel_size, stride
#             nn.Conv2d(6, 16, 5),
#             nn.BatchNorm2d(16),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2),
#             d2l.FlattenLayer(),
#             nn.Linear(16*4*4, 120),
#             nn.BatchNorm1d(120),
#             nn.Sigmoid(),
#             nn.Linear(120, 84),
#             nn.BatchNorm1d(84),
#             nn.Sigmoid(),
#             nn.Linear(84, 10)
#         )

"""


"""
# 残差网络
class Residual(nn.Module):
  def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
    super(Residual, self).__init__()
    if use_1x1conv:
      self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
    else:
        self.conv3 = None
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
  def forward(self, x):
    Y = F.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))
    if self.conv3: X = self.conv3(X)
    return F.relu(Y + X)
"""

"""
# 稠密连接网络（DenseNet）
# 与ResNet的主要区别在于，DenseNet里模块BB的输出不是像ResNet那样和模块AA的输出相加，而是在通道维上连结。
# 这样模块AA的输出可以直接传入模块BB后面的层

def conv_block(in_channels, out_channels):
  blk = nn.Sequential(
    nn.BatchNorm2d(in_channels), 
    nn.ReLU(),
    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
  )
  return blk

# 稠密块由多个conv_block组成，每块使用相同的输出通道数。
# 但在前向计算时，我们将每块的输入和输出在通道维上连结。

class DenseBlock(nn.Module):
  def __init__(self, num_convs, in_channels, out_channels):
    super(DenseBlock, self).__init__()
    net = []
    for i in range(num_convs):
      in_c = in_channels+i*out_channels
      net.append(conv_block(in_c, out_channels))
    self.net = nn.ModuleList(net)
    self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数
  def forward(self, x):
    for blk in self.net:
      y = blk(x) 
      x = torch.cat((x,y),dim=1)
    return x
    
# blk = DenseBlock(2, 3, 10)
# X = torch.rand(4, 3, 8, 8)
# Y = blk(X)
# print (blk)
# print (Y.shape) #torch.Size([4, 23, 8, 8])

# 过渡层
# 由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制模型复杂度。
# 它通过1×11×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。

def transition_block(in_channels, out_channels):
  blk = nn.Sequential(
    nn.BatchNorm2d(in_channels), 
    nn.ReLU(),
    nn.Conv2d(in_channels, out_channels, kernel_size=1),
    nn.AvgPool2d(kernel_size=2, stride=2))
  return blk

# 我们来构造DenseNet模型。DenseNet首先使用同ResNet一样的单卷积层和最大池化层。
net = nn.Sequential(
  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
  nn.BatchNorm2d(64), 
  nn.ReLU(),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  )
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
class FlattenLayer(nn.Module):
  def __init__(self):
    super(FlattenLayer, self).__init__()
  def forward(self, x):
    return x.view(x.shape[0],-1)
# 类似于ResNet接下来使用的4个残差块，DenseNet使用的是4个稠密块。同ResNet一样，我们可以设置每个稠密块使用多少个卷积层。
# 这里我们设成4，从而与上一节的ResNet-18保持一致。稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。
num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]
for i, num_convs in enumerate(num_convs_in_dense_blocks):
  DB = DenseBlock(num_convs, num_channels, growth_rate)
  net.add_module("DenseBlosk_%d" % i, DB)
  # 上一个稠密块的输出通道数
  num_channels = DB.out_channels
  # 在稠密块之间加入通道数减半的过渡层
  if i != len(num_convs_in_dense_blocks) - 1:
    net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
    num_channels = num_channels // 2
net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10))) 
X = torch.rand((1, 1, 96, 96))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)


# 0  output shape:         torch.Size([1, 64, 48, 48])
# 1  output shape:         torch.Size([1, 64, 48, 48])
# 2  output shape:         torch.Size([1, 64, 48, 48])
# 3  output shape:         torch.Size([1, 64, 24, 24])
# DenseBlosk_0  output shape:      torch.Size([1, 192, 24, 24])
# transition_block_0  output shape:        torch.Size([1, 96, 12, 12])
# DenseBlosk_1  output shape:      torch.Size([1, 224, 12, 12])
# transition_block_1  output shape:        torch.Size([1, 112, 6, 6])
# DenseBlosk_2  output shape:      torch.Size([1, 240, 6, 6])
# transition_block_2  output shape:        torch.Size([1, 120, 3, 3])
# DenseBlosk_3  output shape:      torch.Size([1, 248, 3, 3])
# BN  output shape:        torch.Size([1, 248, 3, 3])
# relu  output shape:      torch.Size([1, 248, 3, 3])
# global_avg_pool  output shape:   torch.Size([1, 248, 1, 1])
# fc  output shape:        torch.Size([1, 10])

"""

# **********************************************计算机视觉*************************************************

from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models


# import sys
# sys.path.append("..") 
# import d2lzh_pytorch as d2l
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
#     Y = [aug(img) for _ in range(num_rows * num_cols)]
#     show_images(Y, num_rows, num_cols, scale)

# apply(img, torchvision.transforms.RandomHorizontalFlip())
# apply(img, torchvision.transforms.RandomVerticalFlip())
# # 在下面的代码里，我们每次随机裁剪出一块面积为原面积10%∼100%10%∼100%的区域，
# # 且该区域的宽和高之比随机取自0.5∼20.5∼2，然后再将该区域的宽和高分别缩放到200像素。
# # 若无特殊说明，本节中a和b之间的随机数指的是从区间[a,b]中随机均匀采样所得到的连续值。
# shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)


# 变化颜色
# 另一类增广方法是变化颜色。我们可以从4个方面改变图像的颜色：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
# 在下面的例子里，我们将图像的亮度随机变化为原图亮度的50%（1−0.5）∼150%（1+0.5）。
# apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
# apply(img, torchvision.transforms.ColorJitter(hue=0.5))
# apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

# 我们也可以同时设置如何随机变化图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
# color_aug = torchvision.transforms.ColorJitter(
#   brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

# 叠加多个图像增广方法
# augs = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)

# flip_aug = torchvision.transforms.Compose([
#      torchvision.transforms.RandomHorizontalFlip(),
#      torchvision.transforms.ToTensor()])

# no_aug = torchvision.transforms.Compose([
#      torchvision.transforms.ToTensor()])

# num_workers = 0 if sys.platform.startswith('win32') else 4
# def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
#   dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
#   return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

# 获取数据集
# data_dir = '/S1/CSCL/tangss/Datasets'
# os.listdir(os.path.join(data_dir, "hotdog")) # ['train', 'test']
# # 我们创建两个ImageFolder实例来分别读取训练数据集和测试数据集中的所有图像文件。
# train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
# test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))
# # 指定RGB三个通道的均值和方差来将图像通道归一化
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# train_augs = transforms.Compose([
#   transforms.RandomResizedCrop(size=224),
#   transforms.RandomHorizontalFlip(),
#   transforms.ToTensor(),
#   normalize])

# test_augs = transforms.Compose([
#   transforms.Resize(size=256),
#   transforms.CenterCrop(size=224),
#   transforms.ToTensor(),
#   normalize])

# def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
#   train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
#   batch_size, shuffle=True)
#   test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
#   batch_size)
#   loss = torch.nn.CrossEntropyLoss()
#   train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

"""
# anchor

import math
from matplotlib import pyplot as plt
def bbox_to_rect(bbox,color):
  # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
  # ((左上x, 左上y), 宽, 高)
  return plt.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],fill=False,edgecolor=color,linewidth=2)

img = Image.open('/Users/admin/Desktop/timg.jpg')
w, h = img.size
# print("w = %d, h = %d" % (w, h)) #w = 640, h = 455
def MultiBoxPrior(feature_map, sizes=[0.75,0.5,0.25],ratios=[1,2,0.5]):
  pairs = []
  for r in ratios:
    pairs.append([sizes[0], math.sqrt(r)])
  for s in sizes[1:]:
    pairs.append([s, math.sqrt(ratios[0])])
  
  pairs = np.array(pairs)
  ss1 = pairs[:,0] * pairs[:,1]
  ss2 = pairs[:,0] / pairs[:,1]
  base_anchors = np.stack([-ss1,-ss2,ss1,ss2],axis=1)/2
  
  h, w = feature_map.shape[-2:]
  shifts_x = np.arange(0,w) / w
  shifts_y = np.arange(0,h) / h
  shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
  shift_x = shift_x.reshape(-1)
  shift_y = shift_y.reshape(-1)
  shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
  anchors = shifts.reshape((-1,1,4))+base_anchors.reshape((1,-1,4))
  return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)

X = torch.Tensor(1,3,h,w)
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# print(Y.shape) #torch.Size([1, 1456000, 4])
boxes = Y.reshape((h,w,5,4))

def show_bboxes(axes, bboxes, labels=None, colors=None):
  def _make_list(obj, default_values = None):
    if obj is None:
      obj = default_values
    elif not isinstance(obj, (list,tuple)):
      obj = [obj]
    return obj
  labels = _make_list(labels)
  colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
  for i, bbox in enumerate(bboxes):
    color = colors[i % len(colors)]
    rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
    axes.add_patch(rect)
    if labels and len(labels) > i:
      text_color = 'k' if color == 'w' else 'w'
      axes.text(rect.xy[0], rect.xy[1], labels[i],
      va='center', ha='center', fontsize=6, color=text_color,
      bbox=dict(facecolor=color, lw=0))

# fig = plt.imshow(img)
# bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
# show_bboxes(fig.axes, boxes[225, 320, :, :] * bbox_scale,
# ['s=0.75, r=1', 's=0.75, r=2', 's=0.55, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])
# plt.show()
"""
# jaccard系数
def compute_intersection(set_1, set_2):
  """
  计算anchor之间的交集
    Args:
    set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
    set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
    intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
  """
  lower_bounds = torch.max(set_1[:,:2].unsqueeze(1),set_2[:,2].unsqueeze(0))
  upper_bounds = torch.min(set_1[:,2:].unsqueeze(1),set_2[:,2:].unsqueeze(0))
  intersection_dims = torch.clamp(upper_bounds-lower_bounds,min=0)
  return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]

def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

# set_1 = torch.tensor([[1,2,11,12]])
# set_2 = torch.tensor([[3,6,18,21]])

# print(torch.max(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0)))
# print (set_1[:,:2].unsqueeze(1))
# print (set_2[:,:2].unsqueeze(1))

# 给锚框分配与其相似的真实边界框
# bbox_scale = torch.tensor((w,h,w,h), dtype=torch.float32)
# ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],[1, 0.55, 0.2, 0.9, 0.88]]) 
# anchors = torch.tensor([
#   [0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
#   [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],[0.57, 0.3, 0.92, 0.9]])

# fig = plt.imshow(img)
# show_bboxes(fig.axes, ground_truth[:,1:]*bbox_scale, ['dog','cat'],'k')
# show_bboxes(fig.axes, anchors*bbox_scale,['0', '1', '2', '3', '4'])
# plt.show()

def assign_anchor(bb, anchor, jaccard_threshold=0.5):
  """
  anchor表示成归一化(xmin, ymin, xmax, ymax).
  Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
  Returns:
  assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
  """
  na = anchor.shape[0]
  nb = bb.shape[0]
  jaccard = compute_jaccard(anchor,bb).detach().cpu().numpy() #shape:(na,nb)
  assigned_idx = np.ones(na)*-1 #初始化为-1
  # 先为每个bb分配一个anchor(不要求满足jaccard_threshold)
  jaccard_cp = jaccard.copy()
  for j in range(nb):
    i = np.argmax(jaccard_cp[:,j])
    assigned_idx[i] = j
    jaccard_cp[i, :] = float("-inf") # 赋值为负无穷, 相当于去掉这一行
  
  # 处理还未被分配的anchor，要求满足jaccard_threshold
  for i in range(na):
    if assigned_idx[i] == -1:
      j = np.argmax(jaccard[i,:])
      if jaccard[i,j]>=jaccard_threshold:assigned_idx[i]=j
  return torch.tensor(assigned_idx, dtype=torch.long) #

def xy_to_cxcy(xy):
  """
  将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
  Args:
    xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
  Returns: 
    bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
  """
  return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
  xy[:, 2:] - xy[:, :2]], 1)  # w, h


# xy = torch.tensor([[1,2,3,4],[5,6,7,8]])
# print(xy_to_cxcy(xy))
# tensor([[2, 3, 2, 2],
#         [6, 7, 2, 2]])


def MultiBoxTarget(anchor, label):
  """
  Args:
    anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
    label: 
      真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
      第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
  Returns:
    列表, [bbox_offset, bbox_mask, cls_labels]
    bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
    bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
    cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
  """
  assert len(anchor.shape) == 3 and len(label.shape) == 3
  bn = label.shape[0]
  def MultiBoxTarget_one(anc, lab, eps=1e-6):
    """
    MultiBoxTarget函数的辅助函数, 处理batch中的一个
    Args:
      anc: shape of (锚框总数, 4)
      lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
      eps: 一个极小值, 防止log0
    Returns:
      offset: (锚框总数*4, )
      bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
      cls_labels: (锚框总数, 4), 0代表背景
    """
    an = anc.shape[0]
    assigned_idx = assign_anchor(lab[:,1:],anc) # (锚框总数, )
    bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4) # (锚框总数, 4)
    cls_labels = torch.zeros(an,dtype=torch.long) #0表示背景
    assigned_bb = torch.zeros((an,4),dtype=torch.float32)
    for i in range(an):
      bb_idx = assigned_idx[i]
      if bb_idx>=0:
        cls_labels[i] = lab[bb_idx, 0].long().item() + 1
        assigned_bb[i, :] = lab[bb_idx, 1:]
    center_anc = xy_to_cxcy(anc)
    center_assigned_bb = xy_to_cxcy(assigned_bb)
    offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
    offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim = 1) * bbox_mask # (锚框总数, 4)
    return offset.view(-1), bbox_mask.view(-1), cls_labels
  batch_offset = []
  batch_mask = []
  batch_cls_labels = []
  for b in range(bn):
    offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0,:,:],label[b,:,:])
    batch_offset.append(offset)
    batch_mask.append(bbox_mask)
    batch_cls_labels.append(cls_labels)
  bbox_offset = torch.stack(batch_offset)
  bbox_mask = torch.stack(batch_mask)
  cls_labels = torch.stack(batch_cls_labels)
  return [bbox_offset, bbox_mask, cls_labels]

# 我们通过unsqueeze函数为锚框和真实边界框添加样本维。

# labels = MultiBoxTarget(anchors.unsqueeze(dim=0),ground_truth.unsqueeze(dim=0))
# from collections import namedtuple
# Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])

def non_max_suppression(bb_info_list, nms_threshold=0.5):
  """
    非极大抑制处理预测的边界框
    Args:
        bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
        nms_threshold: 阈值
    Returns:
        output: Pred_BB_Info的列表, 只保留过滤后的边界框信息
  """
  output = []
  # 先根据置信度从高到低排序
  sorted_bb_info_list = sorted(bb_info_list,key=lambda x: x.confidence,reverse=True)
  while len(sorted_bb_info_list)!=0:
    best = sorted_bb_info_list.pop(0)
    output.append(best)
    if len(sorted_bb_info_list) == 0: break
    bb_xyxy = []
    for bb in sorted_bb_info_list:
      bb_xyxy.append(bb.xyxy)
    iou = compute_jaccard(torch.tensor([best.xyxy]),torch.tensor(bb_xyxy))[0]
    n = len(sorted_bb_info_list)
    sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]
  return output

def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold = 0.5):
  """
    Args:
        cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)
        anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
  """
  assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
  bn = cls_prob.shape[0]
  def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold = 0.5):
    """
      MultiBoxDetection的辅助函数, 处理batch中的一个
      Args:
        c_p: (预测总类别数+1, 锚框个数)
        l_p: (锚框个数*4, )
        anc: (锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
      Return:
          output: (锚框个数, 6)
    """
    pred_bb_num = c_p.shape[1]
    anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy() # 加上偏移量
    confidence, class_id = torch.max(c_p, 0)
    confidence = confidence.detach().cpu().numpy()
    class_id = class_id.detach().cpu().numpy()
    pred_bb_info = [Pred_BB_Info(
      index = i,
      class_id = class_id[i] - 1, # 正类label从0开始
      confidence = confidence[i],
      xyxy=[*anc[i]]) # xyxy是个列表
      for i in range(pred_bb_num)]
    # 正类的index
    obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]
    output = []
    for bb in pred_bb_info:
      output.append([
        (bb.class_id if bb.index in obj_bb_idx else -1.0),
        bb.confidence,
        *bb.xyxy])
    return torch.tensor(output) # shape: (锚框个数, 6)
    batch_output = []
    for b in range(bn):
      batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))
    return torch.stack(batch_output)

# output = MultiBoxDetection(cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),anchors.unsqueeze(dim=0), nms_threshold=0.5)

"""
# 构建目标检测数据集
# --pikachu
#   --train
#     --images
#       --1.png
#       ...
#     --label.json
#   --val
#     --images
#       --1.png
#       ...
#     --label.json 


data_dir = '../../data/pikachu'

class PikachuDetDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, part, image_size=(256,256)):
    assert part in ['train', 'val']
    self.image_size = image_size
    self.image_dir = os.path.join(data_dir, part, 'images')

    with open(os.path.join(data_dir, part, 'label.json')) as f:
      self.label = json.load(f)
    
    self.transform = torchvision.transforms.Compose([
      # 将 PIL 图片转换成位于[0.0, 1.0]的floatTensor, shape (C x H x W)
      torchvision.transforms.ToTensor()
    ])
  
  def __len__(self):
    return len(self.label)
  
  def __getitem__(self, index):
    image_path = str(index+1)+'.png'
    image_cls = self.label[image_path]['class']
    label = np.array([image_cls]+self.label[image_path]['loc'],dtype='float32')[None,:]
    pil_img = Image.open(os.path.join(self.image_dir,image_path)).convert('RGB').resize(self.image_size)
    img = self.transform(pil_img)

    sample = {
      'label':label,
      'image':img
    }
    return sample

def load_data_pikachu(batch_size, edge_size=256, data_dir = '../../data/pikachu'):
  image_size = (edge_size,edge_size)
  train_dataset = PikachuDetDataset(data_dir, 'train', image_size)
  val_dataset = PikachuDetDataset(data_dir, 'val', image_size)

  train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  return train_iter, val_iter

batch = iter(train_iter).next()
print(batch["image"].shape, batch["label"].shape)

"""


"""
# SSD

import json
import time
from tqdm import tqdm
from PIL import Image

def cls_predictor(input_channels, num_anchors, num_classes):
  return nn.Conv2d(in_channels=input_channels,out_channels=num_anchors*(num_classes+1),kernel_size=3,padding=1)

def bbox_predictor(input_channels,num_anchors):
  return nn.Conv2d(in_channels=input_channels, out_channels=num_anchors*4, kernel_size=3,padding=1)


def forward(x, block):
  return block(x)

# we can convert the prediction results to binary format (batch size, height, width, number of channels) 
# to facilitate subsequent concatenation on the 1st dimension.
def flatten_pred(pred):
  return pred.permute(0,2,3,1).reshape(pred.size(0), -1)

def concat_preds(preds):
  return torch.cat(tuple([flatten_pred(p) for p in preds]), dim=1)

Y1 = forward(torch.zeros((2,8,20,20)),cls_predictor(8,5,10))
Y2 = forward(torch.zeros((2,16,10,10)),cls_predictor(16,3,10))
# print(concat_preds([Y1,Y2]).shape) #[2,25300]

def down_sample_blk(input_channels, num_channels):
  blk = []
  for _ in range(2):
    blk.append(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3,padding=1))
    blk.append(nn.BatchNorm2d(num_features=num_channels))
    blk.append(nn.ReLU())
    input_channels = num_channels
  blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
  blk = nn.Sequential(*blk)
  return blk

def base_net():
  blk = []
  num_filters = [3, 16, 32, 64]
  for i in range(len(num_filters)-1):
    blk.append(down_sample_blk(num_filters[i],num_filters[i+1]))
  blk = nn.Sequential(*blk)
  return blk

# print (forward(torch.zeros((2, 3, 256, 256)), base_net()).shape) #torch.Size([2, 64, 32, 32])

# print(base_net())
# Sequential(
#   (0): Sequential(
#     (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU()
#     (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU()
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (1): Sequential(
#     (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU()
#     (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU()
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (2): Sequential(
#     (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU()
#     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU()
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
# )

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

import itertools
import math

def create_anchors(feature_map_sizes, steps, sizes):
  scale = 256.
  steps = [s / scale for s in steps]
  sizes = [s / scale for s in sizes]
  aspect_ratios = ((2,),)
  num_layers = len(feature_map_sizes)
  boxes = []
  for i in range(num_layers):
    fmsize = feature_map_sizes[i]
    for h,w in itertools.product(range(fmsize),repeat=2):
      cx = (w+0.5) * steps[i]
      cy = (h+0.5) * steps[i]
      s = sizes[i]
      boxes.append((cx,cy,s,s))
      
      s = sizes[i+1]
      boxes.append((cx,cy,s,s))

      s = sizes[i]
      for ar in aspect_ratios[i]:
        boxes.append((cx,cy,(s * math.sqrt(ar)),(s / math.sqrt(ar))))
        boxes.append((cx,cy,(s / math.sqrt(ar)),(s * math.sqrt(ar))))
  return torch.Tensor(boxes)

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
  Y = blk(X)
  anchors = create_anchors((Y.size(2),), (256/Y.size(2),), size)
  cls_preds = cls_predictor(Y)
  bbox_preds = bbox_predictor(Y)
  return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2*256, 0.272*256], [0.37*256, 0.447*256], [0.54*256, 0.619*256],
         [0.71*256, 0.79*256], [0.88*256, 0.961*256]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
# print(num_anchors) #4

class TinySSD(nn.Module):
  def __init__(self, input_channels, num_classes):
    super(TinySSD,self).__init__()
    input_channels_cls = 128
    input_channels_bbox = 128
    self.num_classes = num_classes
    self.blk = []
    self.cls = []
    self.bbox = []
    
    self.blk_0 = get_blk(0)
    self.blk_1 = get_blk(1)
    self.blk_2 = get_blk(2)
    self.blk_3 = get_blk(3)
    self.blk_4 = get_blk(4)

    self.cls_0 = cls_predictor(64,num_anchors,num_classes)
    self.cls_1 = cls_predictor(input_channels_cls, num_anchors, num_classes)
    self.cls_2 = cls_predictor(input_channels_cls, num_anchors, num_classes)
    self.cls_3 = cls_predictor(input_channels_cls, num_anchors, num_classes)
    self.cls_4 = cls_predictor(input_channels_cls, num_anchors, num_classes)

    self.bbox_0 = bbox_predictor(64,num_anchors)
    self.bbox_1 = bbox_predictor(input_channels_bbox,num_anchors)
    self.bbox_2 = bbox_predictor(input_channels_bbox,num_anchors)
    self.bbox_3 = bbox_predictor(input_channels_bbox,num_anchors)
    self.bbox_4 = bbox_predictor(input_channels_bbox,num_anchors)
  def forward(self, X):
    anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
    X, anchors[0],cls_preds[0],bbox_preds[0] = blk_forward(X,self.blk_0,sizes[0],ratios[0],self.cls_0,self.bbox_0)
    X, anchors[1],cls_preds[1],bbox_preds[1] = blk_forward(X,self.blk_1,sizes[1],ratios[1],self.cls_1,self.bbox_1)
    X, anchors[2],cls_preds[2],bbox_preds[2] = blk_forward(X,self.blk_2,sizes[2],ratios[2],self.cls_2,self.bbox_2)
    X, anchors[3],cls_preds[3],bbox_preds[3] = blk_forward(X,self.blk_3,sizes[3],ratios[3],self.cls_3,self.bbox_3)
    X, anchors[4],cls_preds[4],bbox_preds[4] = blk_forward(X,self.blk_4,sizes[4],ratios[4],self.cls_4,self.bbox_4)
    return (torch.cat(anchors, dim=0), concat_preds(cls_preds).reshape((-1, 5444, self.num_classes + 1)), 
    concat_preds(bbox_preds))

def init_weights(m):
  if type(m) == nn.Linear or type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)

# net = TinySSD(3, num_classes=1)
# net.apply(init_weights)

# X = torch.zeros((32, 3, 256, 256))
# anchors, cls_preds, bbox_preds = net(X)

# print('output anchors:', anchors.shape) # torch.Size([5444, 4])
# print('output class preds:', cls_preds.shape) # torch.Size([32, 5444, 2])
# print('output bbox preds:', bbox_preds.shape) # torch.Size([32, 21776])

# d2l.download_and_preprocess_data()
# batch_size = 32
# data_dir = '../data/pikachu/'
# train_dataset = PikachuDetDataset(data_dir, 'train')
# val_dataset = PikachuDetDataset(data_dir, 'val')

# class PikachuDetDataset(torch.utils.data.Dataset):
#   def __init__(self, data_dir, part, image_size=(256,256)):
#     assert part in ['train', 'val']
#     self.image_size = image_size
#     self.image_dir = os.path.join(data_dir, part, 'images')

#     with open(os.path.join(data_dir, part, 'label.json')) as f:
#       self.label = json.load(f)
    
#     self.transform = torchvision.transforms.Compose([
#       # 将 PIL 图片转换成位于[0.0, 1.0]的floatTensor, shape (C x H x W)
#       torchvision.transforms.ToTensor()
#     ])
  
#   def __len__(self):
#     return len(self.label)
  
#   def __getitem__(self, index):
#     image_path = str(index+1)+'.png'
#     image_cls = self.label[image_path]['class']
#     label = np.array([image_cls]+self.label[image_path]['loc'],dtype='float32')[None,:]
#     pil_img = Image.open(os.path.join(self.image_dir,image_path)).convert('RGB').resize(self.image_size)
#     img = self.transform(pil_img)

#     sample = {
#       'label':label,
#       'image':img
#     }
#     return sample

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# print(device)

net = TinySSD(3,num_classes=1)
net.apply(init_weights)
net = net.to(device)

learning_rate = 1e-3
weight_decay = 5e-4
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

id_cat = dict()
id_cat[0] = 'pikachu'

class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=2, device='cuda:0', eps=1e-10):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.device = device
    self.eps = eps
  def forward(self, input, target):
    p = torch.sigmoid(input)
    pt = p * target.float()+(1.0-p)*(1-target).float()
    alpha_t = (1.0 - self.alpha) * target.float() + self.alpha * (1 - target).float()
    loss = - 1.0 * torch.pow((1 - pt), self.gamma) * torch.log(pt + self.eps)
    return loss.sum()

class SSDLoss(nn.Module):
  def __init__(self, loc_factor, jaccard_overlap, device='cuda:0', **kwargs):
    super().__init__()
    self.f1 = FocalLoss(**kwargs)
    self.loc_factor = loc_factor
    self.jaccard_overlap = jaccard_overlap
    self.device = device
  
  def one_hot_encoding(labels, num_classes):
    return torch.eye(num_classes)[labels]
  
  def loc_transformation(x, anchors, overlap_indicies):
    # Doing location transformations according to SSD paper
    return torch.cat([
      (x[:, 0:1] - anchors[overlap_indicies, 0:1]) / anchors[overlap_indicies, 2:3],
      (x[:, 1:2] - anchors[overlap_indicies, 1:2]) / anchors[overlap_indicies, 3:4],
      torch.log((x[:, 2:3] / anchors[overlap_indicies, 2:3])),
      torch.log((x[:, 3:4] / anchors[overlap_indicies, 3:4]))], dim=1)
  
  def forward(self, class_hat, bb_hat, class_true, bb_true, anchors):
    loc_loss = 0.0
    class_loss = 0.0

    for i in range(len(class_true)):
      # batch_level
      class_hat_i = class_hat[i,:,:]
      bb_true_i = bb_true[i].float()
      class_true_i = class_true[i]
      class_target = torch.zeros(class_hat_i.shape[0]).long().to(self.device)
      overlap_list = assign_anchor(bb_true_i.squeeze(0), anchors, self.jaccard_overlap)
      temp_loc_loss = 0.0
      for j in range((len(overlap_list))):
        overlap = overlap_list[j]
        class_target[overlap]=class_true_i[0,j]
        input_ = bb_hat[i,overlap,:]
        target_ = SSDLoss.loc_transformation(bb_true_i[0, j, :].expand((len(overlap), 4)), anchors, overlap)
        temp_loc_loss += F.smooth_l1_loss(input=input_, target=target_, reduction="sum") / len(overlap)
      loc_loss += temp_loc_loss / class_true_i.shape[1]
      class_target = SSDLoss.one_hot_encoding(class_target, len(id_cat) + 1).float().to(self.device)
      class_loss += self.fl(class_hat_i, class_target) / class_true_i.shape[1]
    loc_loss = loc_loss / len(class_true)
    class_loss = class_loss / len(class_true)
    loss = class_loss + loc_loss * self.loc_factor
    return loss, loc_loss, class_loss

loss = SSDLoss(loc_factor=5.0, jaccard_overlap=0.5, device="cuda:0")

num_epochs = 25
init_epoch = 0

animator = d2l.Animator(xlabel='epoch', xlim=[init_epoch+1, num_epochs],
                        legend=['class error', 'bbox mae', 'train_err'])
                        
for epoch in range(init_epoch, num_epochs):
        
    net.train()
    
    train_loss = 0.0
    loc_loss = 0.0
    class_loss = 0.0

    for i, (x, bb_true, class_true) in (enumerate(train_loader)):
        
        x = x.to(device)
        bb_true = bb_true.to(device)
        class_true = class_true.to(device)
        
        timer_start = time.time()
        
        anchors, cls_preds, bbox_preds = net(x)
        
        class_true = [*class_true.reshape((class_true.size(0), 1, 1))]
        bb_true = [*bb_true.reshape((bb_true.size(0), 1, 1, 4))]
        
        bbox_preds = bbox_preds.reshape((-1, 5444, 4))
        
        # Label the category and offset of each anchor box
        
        anchors = anchors.to(device)
                
        batch_loss, batch_loc_loss, batch_class_loss = loss(cls_preds, bbox_preds, class_true, bb_true, anchors)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        class_loss += batch_class_loss
        loc_loss += batch_loc_loss
        train_loss += batch_loss
t   train_loss = (train_loss/len(train_loader)).detach().cpu().numpy()
    loc_loss = (loc_loss/len(train_loader)).detach().cpu().numpy()
    class_loss = (class_loss/len(train_loader)).detach().cpu().numpy()
    
#     print(class_loss, loc_loss, train_loss, epoch+1)
    
    # Uncomment the following if you wish to see the results after every epoch to see the learning effect
    # Images will be saved to the 'results_every_epoch' directory
    
#     try:
#         d2l.infer(net, epoch, 0.9, device)
#     except Exception as e:
#         print(e, 'error' + str(epoch+1))
        
    
    animator.add(epoch, (class_loss, loc_loss, train_loss))

"""


"""
# R-CNN 系列：R-CNN/FAST R-CNN/FASTER R-CNN/MASK R-CNN
"""





















    






























