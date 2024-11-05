import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore')
def int_Simpson(f,xl:np.array):
    a=xl[0]
    b=xl[-1]
    if len(xl)%2==0:
        raise ValueError('xl must have an odd number of elements')
    n=int((len(xl)-1)/2)
    c=np.ones(2*n+1)
    c[1::2] = 4
    c[0::2] = 2
    c[0]=1
    c[-1]=1
    intfx=(np.dot(f(xl),c))*(b-a)/(6*n)
    return intfx


def R(r: float, Z: float=14) -> float:
    n = 3
    rho = (2 * Z * r) / n
    radial_wave_function = (1 / (9 * np.sqrt(3))) * (6 - 6 * rho + rho**2) * (Z**(3/2)) * np.exp(-rho / 2)
    return radial_wave_function

def R2(r:float,Z:float=14)-> float:
    return R(r,Z)**2*r**2

def R2_ununiform(t):
    r=r0*(np.exp(t)-1)
    return R2(r)*(r+r0)

def p(r,rmax=40):
    tmax=np.log(rmax/r0+1)
    return 1/(r+r0)/tmax




if __name__=='__main__':
    rmax=40
    r0=0.0005
    tmax=np.log(rmax/r0+1)
    intfl=[]
    ml=np.arange(5,10000)
    for m in ml:
        n=2*m+1
        intf=int_Simpson(R2,np.linspace(0,50,n))
        intfl.append(intf)
    intfl1=[]
    ml1=np.arange(5,10000)
    for m in ml1:
        n=2*m+1
        intf=int_Simpson(R2_ununiform,np.linspace(0,tmax,n))
        intfl1.append(intf)

    fig=plt.figure()
    plt.scatter(np.log10(2*ml1+1), np.log10(np.abs(np.array(intfl1) - 1)),s=1,marker="s")
    plt.scatter(np.log10(2*ml+1), np.log10(np.abs(np.array(intfl) - 1)),s=1,marker="s")
    plt.xlabel('log(N)')
    plt.ylabel('log(Error)')
    plt.legend(["Uniform grid sampling","Ununiform grid sampling"])
    plt.title('Convergence of the integral')
    fig.savefig('integrate.png',dpi=600)

    fig=plt.figure(figsize=(12,5))
    plt.subplot(121)
    x=np.linspace(0.0,6,10000)
    y=R2(x)
    plt.plot(x,y)
    plt.ylim(0,1.5)
    plt.xlabel('r')
    plt.ylabel(r'y')
    plt.legend([r'$\|R_{3s}\|^2r^2$'])
    plt.title(r'(a)$\|R_{3s}\|^2r^2$')
    plt.subplot(122)
    x=np.linspace(0.0,tmax,10000)
    y=(np.exp(x)-1)*r0
    plt.hist(y,bins=100,density=True); 
    plt.plot(y,p(y))
    plt.ylim(0,0.5)
    plt.title('(b)Probability of r(t)')
    plt.xlabel('r')
    plt.ylabel('p(r)')
    plt.legend(['p(r)','Histogram of r0 * (exp(t)-1)'])
    
    fig.savefig('R2.png',dpi=1000)

    print(f"Simpson Intergrate with uniform grid(N=1001):{intfl[495]}")
    print(f"Simpson Intergrate with ununiform grid(N=1001):{intfl1[495]}")
    
