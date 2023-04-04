#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycaret

#%%
import torch
# 뉴럴네트워크
import torch.nn as nn
# 뉴럴네트워크용 함수들
import torch.nn.functional as nf
# 옵티마이즈
import torch.optim as optim

# %%
x=[1,2,3]
y=[2,4,8]
plt.plot(x,y,'o-')
# %%
x_=[[1],[2],[3]]
y_=[[2],[4],[8]] # 실제값

# %%
# 트레인 데이터 준비
x_train=torch.FloatTensor(x_)
y_train=torch.FloatTensor(y_)
print(x_train.shape,y_train.shape)
print(x_train.dtype,y_train.dtype)
# %%
# 모델성정 모델가정
# Y=W*x+b
# Gradient 하게 추적관리할 데이터생성
W=torch.zeros(1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)
print(W,b)
# %%
# 에러설정 -> 옵티마이즈의 기준 MSE
# error, cost, loss
ye=W*x_train+b # model
error=torch.mean((ye-y_train)**2)
error
# %%
# 에러를 최적화 하기위한 옵티마이저 설정
# 비교적 단순모델에 유용한 SGD   / SGD, 아담, R뭐시기 순서
optizer=optim.SGD([W,b],lr=0.01)
optizer
# %%
# SSD를 활용한 반복 주석 epoch
epoch=100
ws=[]
bs=[]
for i in range(epoch):
    # 업데이트 된 W로 예측값 뽑고
    ye=W*x_train+b # model
    # 에러를 계산
    error=torch.mean((ye-y_train)**2)

    # 옵티마이저 초기화(더이상 업데이트할게 없다.)
    optizer.zero_grad()
    # 에러기반으로 업데이트
    error.backward()
    # 전진하여 업데이터
    optizer.step()
    if (i%10==0):
        ws.append(W.item())
        bs.append(b.item())
        print(W,b)
# %%
ws
# %%
plt.plot(list(range(len(ws))),ws)
# %%
plt.plot(bs)
# %%
x=[1,2,3]
y=[2,4,8]
plt.plot(x,y,'o-')

def line(w,b,x):
    return w*x+b

for i in range(len(ws)):

    y1=line(ws[i],bs[i],1)
    y2=line(ws[i],bs[i],3)
    plt.plot([1,3],[y1,y2],label=(str(i)))
plt.legend()
plt.show()

# %%
