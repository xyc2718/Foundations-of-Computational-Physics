k=26.24684351 #nm^-2 eV^-1  2m/h^2
V0=10 #eV
a=0.2 #nm
import numpy as np
import matplotlib.pyplot as plt

def f_even(E):
    return np.sqrt(E)*np.sin(np.sqrt(k*a**2*E))-np.sqrt(V0-E)*np.cos(np.sqrt(k*a**2*E))
def f_odd(E):
    return np.sqrt(E)*np.cos(np.sqrt(k*a**2*E))+np.sqrt(V0-E)*np.sin(np.sqrt(k*a**2*E))

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
def bisection_method(f, rg, tol=1e-4, max_iter=100):
    """
    Bisection method to find the root of the function f
    :param f: function
    :param rg: range    
    :param tol: tolerance
    :param max_iter: maximum iteration
    """
    a,b=rg
    if f(a) * f(b) >= 0:
        raise ValueError("Function does not change sign in the interval.")
    converged=0
    for i in range(max_iter):
        c = (a + b) / 2
        if np.abs(b - a) < tol:
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    else:
        converged=1
    if converged:
        print("The bisection method did not converge after", max_iter, "iterations.")
    return (a + b) / 2,np.abs(b - a)

def wave_even(x,E=1):
    """
    wave function of even parity C=F,A=0
    :param x: position
    :param E: energy
    """
    beta=np.sqrt((V0-E)*k)
    C=F=1
    B=np.e**(-beta*a)/np.cos(np.sqrt(k*a**2*E)) #calculate the coefficientB,C,F
    if x>a:
        return F*np.e**(-beta*x)
    elif x<-a:
        return C*np.e**(beta*x)
    else:
        return B*np.cos(np.sqrt(k*E)*x)
    
def wave_odd(x,E=1):
    """
    wave function of odd parity C=-F,B=0
    :param x: position
    :param E: energy
    """
    beta=np.sqrt((V0-E)*k)
    F=1
    C=-F
    A=np.e**(-beta*a)/np.sin(np.sqrt(k*a**2*E))
    if x>a:
        return F*np.e**(-beta*x)
    elif x<-a:
        return C*np.e**(beta*x)
    else:
        return A*np.sin(np.sqrt(k*E)*x)
    

def int_Trapezoidal(f,a,b,n):
    """
    integrate the function f by Trapezoidal method
    :param f: function
    :param a: lower limit
    :param b: upper limit
    :param n: number of points
    """
    x=np.linspace(a,b,n+1,dtype=np.float64)
    intfx=(np.sum([f(i) for i in x])-1/2*f(x[0])-1/2*f(x[-1]))*(b-a)/n
    return intfx

def print_even(E,i):
    """
    print the expression of wave function of even parity\
    """
    beta=np.sqrt((V0-E)*k)
    C=F=1
    B=np.e**(-beta*a)/np.cos(np.sqrt(k*a**2*E))
    print(f"\nwave function of E{i}={E:.4f}eV:")
    norm=np.sqrt(int_Trapezoidal(lambda x:wave_even(x,E)**2,-10*a,10*a,10000))
    print(f"{F/norm:.{4}f} Exp(-{beta:.3f}*x) for x>a")
    print(f"{C/norm:.{4}f} Exp({beta:.3f}*x) for x<-a")
    print(f"{B/norm:.{4}f} cos({np.sqrt(k*E):.3f}*x) for -a<x<a")
def print_odd(E,i):
    """
    print the expression of wave function of odd parity
    """
    beta=np.sqrt((V0-E)*k)
    F=1
    C=-F
    A=np.e**(-beta*a)/np.sin(np.sqrt(k*a**2*E))
    print(f"\nwave function of E{i}={E:.4f}eV:")
    norm=np.sqrt(int_Trapezoidal(lambda x:wave_odd(x,E)**2,-10*a,10*a,10000))
    print(f"{F/norm:.{4}f} Exp(-{beta:.3f}*x) for x>a")
    print(f"{C/norm:.{4}f} Exp({beta:.3f}*x) for x<-a")
    print(f"{A/norm:.{4}f} sin({np.sqrt(k*E):.3f}*x) for -a<x<a")

fig1=plt.figure(figsize=(10,5))
plt.subplot(121)
bracket_even=find_bracket(f_even,[0,10],n=10,ifplot=True)
plt.xlabel("E(eV)")
plt.ylabel(r"$f_{even}(E)$")
plt.subplot(122)

bracket_odd=find_bracket(f_odd,[0,10],n=10,ifplot=True)
plt.xlabel("E(eV)")
plt.ylabel(r"$f_{odd}(E)$")
fig1.savefig("f_even and f_odd.png",dpi=600)

E1,err1=bisection_method(f_even,bracket_even[0]);
E2,err2=bisection_method(f_odd,bracket_odd[0]);
E3,err3=bisection_method(f_even,bracket_even[1]);
print_even(E1,0)
print_odd(E2,1)
print_even(E3,2)

fig2=plt.figure(figsize=(16,4))
plt.subplot(131)
x=np.linspace(-5*a,5*a,1000)
plt.plot(x,[wave_even(i,E1) for i in x])
plt.title(f"wave function of E0={E1:.4f}eV")
plt.xlabel("x(nm)")
plt.ylabel(r"$\psi_0$")
plt.subplot(132)
x=np.linspace(-5*a,5*a,1000)
plt.plot(x,[wave_odd(i,E2) for i in x])
plt.title(f"wave function of E1={E2:.4f}eV")
plt.xlabel("x(nm)")
plt.ylabel(r"$\psi_1$")
plt.subplot(133)
x=np.linspace(-5*a,5*a,10000)
plt.plot(x,[wave_even(i,E3) for i in x])
plt.title(f"wave function of E2={E3:.4f}eV")
plt.xlabel("x(nm)")
plt.ylabel(r"$\psi_2$")
fig2.savefig("wavefunction.png",dpi=600)
plt.show()