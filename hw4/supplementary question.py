import numpy as np
import matplotlib.pyplot as plt
from gaussian_elimination import *

def f(xl:np.array,yl:np.array)->float:
    if len(xl)!=len(yl):
        raise ValueError("The length of the x and y arrays must be the same")
    if len(xl)<=1:
        return yl[0]
    else:
        return (f(xl[1::],yl[1::])-f(xl[0:-1:],yl[0:-1:]))/(xl[-1]-xl[0])
    
def NewtonCoefficients(xl:np.array,yl:np.array)->np.array:
    return [f(xl[0:n],yl[0:n]) for n in range(1,len(xl)+1)]

def interpolatefunc(x:np.array,xl:np.array,cl:np.array)->np.array:
    y=np.zeros(x.shape)
    for i in range(len(cl)):
        y=y+cl[i]*np.prod([x-xi for xi in xl[:i]],axis=0)
    return y

xl=np.linspace(0,3,10)
yl=1/(1+25*xl**2)
xl1=xl[::-1]
yl1=yl[::-1]
cl=NewtonCoefficients(xl,yl)
cl1=NewtonCoefficients(xl1,yl1)
xx=np.linspace(0,3,1000)
xn=np.array([xx**n for n in range(len(xl))])
fig=plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(xx,interpolatefunc(xx,xl,cl),linewidth=2,alpha=0.5)
plt.plot(xx,interpolatefunc(xx,xl1,cl1),linewidth=2,alpha=0.5)
plt.scatter(xl,yl,marker="s",color="red")
plt.legend(["Forward","Backward","Sample Points"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolating for 1/(1+25x^2)")
plt.subplot(1,2,2)
plt.plot(xx,interpolatefunc(xx,xl1,cl1)-interpolatefunc(xx,xl,cl),linewidth=2,alpha=0.8)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Difference between Forward and Backward")
fig.savefig("newton_interpolation of different order.png",dpi=1000)