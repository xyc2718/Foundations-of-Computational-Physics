import numpy as np
import matplotlib.pyplot as plt
import decimal
decimal.getcontext().prec = 50
def f(x):
    return x**3-5*x+3
def  df(x):
    return 3*x**2-5
def f_decimal(x):
    xd=decimal.Decimal(x)
    return x**3-5*x+3
def  df_decimal(x):
    xd=decimal.Decimal(x)
    return 3*xd**2-5
def find_bracket(f,rg=[-3,3],n=10,ifplot=False):
    """
    find the bracket of the root of the function f in the range rg by sampling n points
    :param f: function
    :param rg: range
    :param n: number of points
    :return: the bracket of the root
    """
    x=np.linspace(rg[0],rg[1],n)
    y=f(x)
    y_roll=np.roll(y,1)
    y_cov=y[1:]*y_roll[1:]
    cr0=np.where(y_cov<0)
    bracket=[(x[i],x[i+1]) for i in cr0[0]]
    if ifplot:
        for i in range(len(bracket)):
            xx=np.linspace(rg[0],rg[1],1000)
            plt.plot(xx,f(xx))
            plt.plot(bracket[i],[0,0],c='r')
            plt.scatter(bracket[i],[0,0],c='b',marker="s")
    return bracket

def bisection_method(f, rg, tol=1e-4, max_iter=100,ifplot=False):
    a,b=rg
    if f(a) * f(b) >= 0:
        raise ValueError("Function does not change sign in the interval.")
    converged=0
    flist=[]
    for i in range(max_iter):
        c = (a + b) / 2
        if np.abs(b - a) < tol:
            print(f"The bisection method converged after {i} iterations")
            print(f"The root is {c}",f"The error is {np.abs(b - a)}")
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
        flist.append(f(c))
    else:
        converged=1
    if converged:
        print("The bisection method did not converge after", max_iter, "iterations.")
    flist=np.array(flist)
    return (a + b) / 2,np.abs(b - a),flist

fig=plt.figure()
find_bracket(f,[-3,3],n=10,ifplot=True)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) with bracket")
plt.legend(["f(x)","bracket"])
plt.show()
fig.savefig("f(x) with bracket.png",dpi=600)