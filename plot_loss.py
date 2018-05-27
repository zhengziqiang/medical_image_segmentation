# import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas
# # from pandas.tools.plotting import  andrews_curves
# from pandas.plotting import  andrews_curves
# data=np.genfromtxt("tmp.csv",dtype=str,delimiter=' ')
# def get_number(string):
#     data=[]
#     for i in range(len(string)):
#         name=string[i]
#         number=filter(lambda ch: ch in '0123456789.', name)
#         number=np.float32(number)
#         data.append(number)
#     return data
# loss_point=data[:,2]
# loss_point_=get_number(loss_point)
# loss_point1=data[:,3]
# loss_point1_=get_number(loss_point1)
# loss_point2=data[:,4]
# loss_point2_=get_number(loss_point2)
# loss_point3=data[:,5]
# loss_point3_=get_number(loss_point3)
#
# x_data=np.linspace(0,3000,num=3000)
# plt.plot(x_data,loss_point_,label="gen_loss_GAN",color="red",linewidth=1)
# plt.plot(x_data,loss_point1_,label="gen_loss_L1",color="black",linewidth=1)
# plt.plot(x_data,loss_point2_,label="dice_loss",color="blue",linewidth=1)
# plt.plot(x_data,loss_point3_,label="discrim_loss",color="yellow",linewidth=1)
# plt.xlabel("The step")
# plt.ylabel("loss")
# plt.title("The loss of training")
# plt.legend()
# plt.savefig("loss.png")
#
# plt.show()


import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas
# from pandas.tools.plotting import  andrews_curves
from pandas.plotting import  andrews_curves
data=np.genfromtxt("tmp.csv",dtype=str,delimiter=' ')
def get_number(string):
    data=[]
    for i in range(len(string)):
        name=string[i]
        number=filter(lambda ch: ch in '0123456789.', name)
        number=np.float32(number)
        data.append(number)
    return data
loss_point=data[:,2]
loss_point_=get_number(loss_point)
loss_point1=data[:,3]
loss_point1_=get_number(loss_point1)
loss_point2=data[:,4]
loss_point2_=get_number(loss_point2)
loss_point3=data[:,5]
loss_point3_=get_number(loss_point3)

x_data=np.linspace(0,3000,num=3000)
plt.plot(x_data,loss_point1_,label="gen_loss_L1",color="red",linewidth=2)
plt.plot(x_data,loss_point2_,label="dice_loss",color="blue",linewidth=1)

plt.xlabel("The step")
plt.ylabel("loss")
plt.title("The loss of training")
plt.legend()
plt.savefig("loss_detail.png")

plt.show()