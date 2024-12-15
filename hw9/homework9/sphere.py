import numpy as np
import math
import matplotlib.pyplot as plt 


def fn(x): #定义圆内函数
    if np.linalg.norm(x) <= 1:
        return 1
    else:
        return 0
def int_mc_n(fn,r,point):  #根据采样点计算蒙特卡洛积分
    return np.mean([fn(p)/r(p) for p in point])
def V0(dim):  #体积的标准值
    return np.pi**(dim/2)/math.gamma(dim/2+1)

def PRN(n,dim,rmin=0,rmax=1,seed=515): #伪随机数生成器
    m = 2**31-1
    a = 48271
    b = 43
    x = seed
    result = []
    for i in range(n*dim):
        x = (a*x+b)%m
        result.append((rmin+(rmax-rmin)*x/m))
    return np.array(result).reshape(n,dim)

if __name__ == '__main__':
    dim=2
    fig=plt.figure(figsize=(12,5))
    y=PRN(10000,2,rmin=-1,rmax=1)
    plt.subplot(121)
    plt.hist(y[:,0],bins=100,density=True);
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Distribution of Pseudorandom Number Generators")
    plt.xlim(-1,1)
    plt.subplot(122)
    plt.scatter(y[:,0],y[:,1],s=1)
    idx=np.array([fn(x) for x in y],dtype=bool)
    plt.scatter(y[idx,0],y[idx,1],s=1,c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Random Sampling for 2D Circle")
    fig.savefig('PRN.png',dpi=600)

    nsample=100000
    fig=plt.figure(figsize=(12,5))
    Vl=[]
    diml=range(1,8)
    for dim in diml:
        point=PRN(nsample,dim,rmin=-1,rmax=1)
        intfx=int_mc_n(fn,lambda x:1/2**dim,point)
        print(f"Dimension:{dim},Volume:{intfx},Standard Volume:{V0(dim)},Sample Points:{nsample}")
        Vl.append(intfx)
    Vl=np.array(Vl)
    Vs=np.array([V0(d) for d in diml])
    plt.subplot(121)
    plt.scatter(diml,Vl,c="r",marker="s");
    plt.plot(diml,Vs,linestyle="--");
    plt.xlabel('dimension');
    plt.ylabel('Volume');
    plt.title('Volume-dimension relation');
    plt.legend(['Standard volume','Monte Carlo integration']);
    plt.subplot(122)
    plt.plot(diml,np.abs(Vl-Vs)/Vs*100,linestyle="--",marker="s");
    plt.xlabel('dimension');
    plt.ylabel('Relative Volume Difference/%');
    plt.title('Relative Volume Difference-dimension difference');
    fig.savefig('VolumeDim.png',dpi=600)
