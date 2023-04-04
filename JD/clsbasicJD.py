#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycaret

#%%
import torch

# %%
def add(a,b):
    print('+',b,'=',a+b)
    return a+b
def sub(a,b):
    print('-',b,'=',a-b)
    return a-b
num=0
num=add(num,3)
num=add(num,2)
num=sub(num,3)


#%%
# 클래스: 함수의 덩어리
# 프로퍼티: 속성
# 매서드: 방법
# 사용: 클래스 선언->매서드와 속성을 가진 빵 들
#       클래스 인스턴트 생성
#       클래스 초기화 -> 속성지정을 통해 같은 모양의 다른 빵을 제작
#                       팥붕어빵, 치즈붕어빵
#
# 클래스 선언
class Calc:
    num=0
    cnt=0
    def __init__(self,n_=0):  ## 생성함수 예약어 
        self.num=n_


    def add(self,b):
        print('+',b,'=',self.num+b)
        self.num=self.num+b
    def sub(self,b):
        print('-',b,'=',self.num-b)
        self.num= self.num-b

# 인스턴스 생성
calc1=Calc()
print ('prop:',calc1.num) # 속성 추출
print (calc1.add(3))
# 초기화를 통한 프로퍼티 변경
calc2=Calc(2)
print ('prop:',calc2.num)
print (calc2.add(3))


# %%
# 상속
class Comp(Calc):
    name="mac1"

notebook=Comp()
print (notebook.name)
print(notebook.num)

print(notebook.add(3))
# %%
# motor(모터달려서 굴러가기) 부모클래스
# wheel 갯수
# wheel 반지름
# 분당 휠 회전속도
# run 함수를 통해 분을 입력하면 얼마를 달렸는지 표시
# 브랜드명

# bike,car 자식클래스
# bike: 앞바퀴 들기() 2/3만 감
#       고속주행:120% 더 감
# car: 
#   물건싣기():
#   적재중량 kg당 1% 속도감소
#
#   bike / car 상속을 이용해 만든 60분간 달리기 했을때 효율을 측정
# %%
class Motor:
    def __init__(self, num_wheels, wheel_radius, rotation_speed, brand):
        self.num_wheels = num_wheels
        self.wheel_radius = 20
        self.rotation_speed = rotation_speed
        self.brand = brand
    
    def run(self, minutes):
        distance = (self.wheel_radius * 2 * 3.14) * self.rotation_speed * minutes
        print(f"{self.brand} 모터가 {distance:.2f}km를 달렸습니다.")
        
        
class Bike(Motor):
    def __init__(self, num_wheels, wheel_radius, rotation_speed, brand):
        super().__init__(num_wheels, wheel_radius, rotation_speed, brand)
        
    def lift_front_wheel(self):
        # 앞바퀴 들기
        self.rotation_speed *= 2/3
    
    def run(self, minutes, is_high_speed=False):
        if is_high_speed:
            self.rotation_speed *= 1.2
        super().run(minutes)
        
        
class Car(Motor):
    def __init__(self, num_wheels, wheel_radius, rotation_speed, brand, max_load_weight):
        super().__init__(num_wheels, wheel_radius, rotation_speed, brand)
        self.max_load_weight = max_load_weight
    
    def load_weight(self, weight):
        self.rotation_speed *= (1 - weight / self.max_load_weight / 100)
    
    def run(self, minutes):
        super().run(minutes)
        
        
# 테스트 코드
bike = Bike(num_wheels=2, wheel_radius=0.5, rotation_speed=50, brand="BikeBrand")
bike.run(minutes=60, is_high_speed=True)
bike.lift_front_wheel()
bike.run(minutes=60, is_high_speed=True)

car = Car(num_wheels=4, wheel_radius=0.7, rotation_speed=70, brand="CarBrand", max_load_weight=1000)
car.run(minutes=60)
car.load_weight(weight=1000)
car.run(minutes=60)



# %%
class mortor:
    rpm=0
    r=0
    wheel=0
    def __init__(self,_w=0,_r=0,_m=0):  # 일반함구화 매서드와의 차이는 self
        self.wheel=_w
        self.r=_r
        self.rpm=_m

    def showMeter(self,min=0):
        dist=min*self.rpm*2*3.14*self.r
        return dist
m1=mortor(4,20,60)
print(m1.showMeter(10))
# %%
class bike(mortor):
    def __init__(self):
        super().__init__(2,10,10)
        self.wheel=2

    def willy(self,min=0):
        return self.showMeter()*0.67
    
    def turbo(self):
        return self.showMeter()*1.2
#   bike :  앞바퀴들기() 2/3만 감
#           고속주행 120% 더 감

class car(mortor):
    def __init__(self):
        super().__init__(4,10,10)
        self.wheel=4

    def logis(self,wgt=0):
        res=0
        if(wgt<100):
            res=self.showMeter()- self.showMeter()*wgt*0.1
        return res
# car
#   물건싣기()
#   적재중량 kg 당 1% 속도감소

c1=car()
c1.logis()
c1.rpm

# %%

# %%
