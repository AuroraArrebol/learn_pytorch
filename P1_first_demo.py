"""
作者：ZWP
日期：2022.07.11
"""
print("列表:")
num=list(range(1,101))
print(f"1.{num}")
#列表函数
print(f"2.{max(num)}")
print(f"3.{min(num)}")
print(f"4.{sum(num)}")
#列表解析
squ=[a**2 for a in num]
print(f"5.{squ}")
#列表切片
print(f"6.{squ[0:10]}")
print(f"7.{squ[0:10:2]}")
print(f"8.{squ[-3:]}")
print(f"9.{squ[::-2]}")

if 1 in num:
    print('y')
if 102 not in num:
    print('n')

#字典
print("\n字典:")
alien={}
alien["X"]=1            #添加键值对
alien["Y"]=2            #添加键值对
alien["Z"]=3            #添加键值对
print(f"alien:{alien}")
del alien['Y']          #删除键值对
print(f"alien:{alien}")
print(f"X:{alien['X']}\n")

favorite={"Tom":"python","Michelle":"C++"}
print(f"favorite:{favorite}\n")


point=alien.get('point','No point value given') #get函数，如果字典中有键，则返回值；无此键则返回第二个参数
print(point)
alien['point']=100
point=alien.get('point','No point value given')
print(point)

#遍历字典（键值对）
for i,j in alien.items():
    print(f"{i}:{j}")

#字典列表
alien_0={"X":1,'Y':2}
alien_1={"X":6,'Y':8}
alien_2={"X":4,'Y':3}
aliens=[alien_0,alien_1,alien_1]
for each in aliens:
     print(each)

#面向对象
class Car:
    def __init__(self,name):
         self.name=name;
    def describe(self):
        print(f"It's a {self.name}")

class ElectroCar(Car):
    def __init__(self,name):
        super().__init__(name)
        self.battery=10
    def describe_battery(self):
        print(f"It's battery size is {self.battery}")

mycar=ElectroCar('Tesla')
mycar.describe()
mycar.describe_battery()



