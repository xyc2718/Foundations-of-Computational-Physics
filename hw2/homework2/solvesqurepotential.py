k=26.24684351 #nm^-2 eV^-1  2m/h^2
V0=10 #eV
a=0.2 #nm
import numpy as np
import matplotlib.pyplot as plt
from findroot import find_bracket,bisection_method
def f_even(E):
    return np.sqrt(E)*np.sin(np.sqrt(k*a**2*E))-np.sqrt(V0-E)*np.cos(np.sqrt(k*a**2*E))
def f_odd(E):
    return np.sqrt(E)*np.cos(np.sqrt(k*a**2*E))+np.sqrt(V0-E)*np.sin(np.sqrt(k*a**2*E))


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

def print_even(E,i,norm):
    """
    print the expression of wave function of even parity\
    """
    beta=np.sqrt((V0-E)*k)
    C=F=1
    B=np.e**(-beta*a)/np.cos(np.sqrt(k*a**2*E))
    print(f"\nwave function of E{i}={E:.5f}eV:")
    print(f"{F/norm:.{5}f} Exp(-{beta:.5f}*x) for x>a")
    print(f"{C/norm:.{5}f} Exp({beta:.5f}*x) for x<-a")
    print(f"{B/norm:.{5}f} cos({np.sqrt(k*E):.5f}*x) for -a<x<a")
def print_odd(E,i,norm):
    """
    print the expression of wave function of odd parity
    """
    beta=np.sqrt((V0-E)*k)
    F=1
    C=-F
    A=np.e**(-beta*a)/np.sin(np.sqrt(k*a**2*E))
    print(f"\nwave function of E{i}={E:.5f}eV:")
    print(f"{F/norm:.{5}f} Exp(-{beta:.5f}*x) for x>a")
    print(f"{C/norm:.{5}f} Exp({beta:.5f}*x) for x<-a")
    print(f"{A/norm:.{5}f} sin({np.sqrt(k*E):.5f}*x) for -a<x<a")

if __name__=="__main__":
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
    #calculate the energy
    E1,err1=bisection_method(f_even,bracket_even[0],tol=1e-8);
    E2,err2=bisection_method(f_odd,bracket_odd[0],tol=1e-8);
    E3,err3=bisection_method(f_even,bracket_even[1],tol=1e-8);


    ##plot wave function and output the expression
    #wave function1
    fig2=plt.figure(figsize=(16,4))
    plt.subplot(131)
    x=np.linspace(-5*a,5*a,1000)
    norm=np.sqrt(int_Trapezoidal(lambda x:wave_even(x,E1)**2,-50*a,50*a,100000))#normalize the wave function
    print_even(E1,0,norm) #output the expression of wave function
    plt.plot(x,[wave_even(i,E1)/norm for i in x])#plot the wave function
    plt.title(f"wave function of E0={E1:.5f}eV")
    plt.xlabel("x(nm)")
    plt.ylabel(r"$\psi_0$")

    #wave function2
    plt.subplot(132)
    x=np.linspace(-5*a,5*a,1000)
    norm=np.sqrt(int_Trapezoidal(lambda x:wave_odd(x,E2)**2,-50*a,50*a,100000))#normalize the wave function
    print_odd(E2,0,norm)#output the expression of wave function
    plt.plot(x,[wave_odd(i,E2)/norm for i in x])#plot the wave function
    plt.title(f"wave function of E1={E2:.5f}eV")
    plt.xlabel("x(nm)")
    plt.ylabel(r"$\psi_1$")

    #wave function3
    plt.subplot(133)
    norm=np.sqrt(int_Trapezoidal(lambda x:wave_even(x,E3)**2,-50*a,50*a,100000)) #normalize the wave function
    print_even(E3,0,norm)#output the expression of wave function
    plt.plot(x,[wave_even(i,E3)/norm for i in x])#plot the wave function
    plt.title(f"wave function of E2={E3:.5f}eV")
    plt.xlabel("x(nm)")
    plt.ylabel(r"$\psi_2$")
    fig2.savefig("wavefunction.png",dpi=600)
    plt.show()