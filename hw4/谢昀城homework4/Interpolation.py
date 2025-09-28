import numpy as np
import matplotlib.pyplot as plt
from gaussian_elimination import *


"""
coefficients of the f[x1,...xn]
"""
def f(xl:np.array,yl:np.array)->float:
    if len(xl)!=len(yl):
        raise ValueError("The length of the x and y arrays must be the same")
    if len(xl)<=1:
        return yl[0]
    else:
        return (f(xl[1::],yl[1::])-f(xl[0:-1:],yl[0:-1:]))/(xl[-1]-xl[0])

"""
Coefficients of the Newton's Interpolating polynomial
"""   
def Newton_coefficients(xl:np.array,yl:np.array)->np.array:
    return [f(xl[0:n],yl[0:n]) for n in range(1,len(xl)+1)]
"""
get Newtown Interpolating function 
"""
def Newton_interpolatefunc(x:np.array,xl:np.array,cl:np.array)->np.array:
    y=np.zeros(x.shape)
    for i in range(len(cl)):
        y=y+cl[i]*np.prod([x-xi for xi in xl[:i]],axis=0)
    return y

"""
Cubic splines interpolation
"""
def splines_funciton(x:float,xl:np.array,yl:np.array)->np.array:
    n=len(xl)
    A=np.zeros((n-2,n-2))
    B=np.zeros(n-2)
    for i in range(1,n-1):
        A[i-1,i-1]=2*(xl[i+1]-xl[i-1])
        if i<n-2:
            A[i-1,i]=xl[i+1]-xl[i]
        if i>1:
            A[i-1,i-2]=xl[i]-xl[i-1]
        if i>1 and i<n-2:
            B[i-1]=6*(yl[i+1]-yl[i])/(xl[i+1]-xl[i])+6*(yl[i-1]-yl[i])/(xl[i]-xl[i-1])

    B[0]=6*(yl[2]-yl[1])/(xl[2]-xl[1])+6*(yl[0]-yl[1])/(xl[1]-xl[0])
    B[-1]=6*(yl[n-1]-yl[n-2])/(xl[n-1]-xl[n-2])+6*(yl[n-3]-yl[n-2])/(xl[n-2]-xl[n-3])
    M=np.hstack([A,B.reshape(-1, 1)])
    M0,c,flag=gaussian_elimination(M)
    cl=np.hstack([0,c,0])
    for i in range(1,n):
        if xl[i-1]<=x and x<=xl[i]:
            return cl[i-1]*(xl[i]-x)**3/(6*(xl[i]-xl[i-1]))+cl[i]*(x-xl[i-1])**3/(6*(xl[i]-xl[i-1]))+(yl[i-1]/(xl[i]-xl[i-1])-cl[i-1]*(xl[i]-xl[i-1])/6)*(xl[i]-x)+(yl[i]/(xl[i]-xl[i-1])-cl[i]*(xl[i]-xl[i-1])/6)*(x-xl[i-1])


if __name__=="__main__":       
    xl=np.linspace(0,np.pi,10)
    yl=np.cos(xl)
    cl=Newton_coefficients(xl,yl)
    xx=np.linspace(0,np.pi,100)
    xn=np.array([xx**n for n in range(len(xl))])
    yspl=np.array([splines_funciton(x,xl,yl) for x in xx])
    fig=plt.figure()
    plt.plot(xx,Newton_interpolatefunc(xx,xl,cl),linewidth=4,alpha=0.5)
    plt.plot(xx,yspl,linewidth=2,alpha=0.5)
    plt.plot(xx,np.cos(xx),linestyle="--")
    plt.scatter(xl,yl,marker="s",color="red")
    plt.legend(["Newton's Interpolating","Cubic Splines Interpolating","Real Function","Sample Points"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolating for cos(x)")
    fig.savefig("Interpolation1.png")
    res1=(Newton_interpolatefunc(xx,xl,cl)-np.cos(xx))**2/len(xx)
    res2=(yspl-np.cos(xx))**2/len(xx)
    print("Interpolation: cos(x)")
    print("Residual of Newton's Interpolating: ",np.sum(res1))
    print("Residual of Cubic Splines Interpolating: ",np.sum(res2))


    xl=np.linspace(-1,1,10)
    yl=1/(1+25*xl**2)
    cl=Newton_coefficients(xl,yl)
    xx0=np.linspace(-1.05,1.05,1000)
    xx=np.linspace(-1.0,1.0,1000)
    xn=np.array([xx**n for n in range(len(xl))  ])
    fig=plt.figure()
    plt.plot(xx0,Newton_interpolatefunc(xx0,xl,cl),linewidth=2,alpha=0.5)
    yspl=np.array([splines_funciton(x,xl,yl) for x in xx])
    plt.plot(xx,yspl,linewidth=2,alpha=0.5)
    plt.plot(xx0,1/(1+25*xx0**2))
    plt.scatter(xl,yl,marker="s",color="red")
    plt.legend(["Newton's Interpolating","Cubic Splines Interpolating","Real Function","Sample Points"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolating for 1/(1+25x^2)")
    fig.savefig("Interpolation2.png")
    print("Interpolation: 1/(1+25x^2)")
    res1=(Newton_interpolatefunc(xx,xl,cl)-1/(1+25*xx**2))**2/len(xx)
    res2=(yspl-1/(1+25*xx**2))**2/len(xx)
    print("Residual of Newton's Interpolating: ",np.sum(res1))
    print("Residual of Cubic Splines Interpolating: ",np.sum(res2))