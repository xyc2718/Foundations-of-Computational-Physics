import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def Sij(Si, Sj):
    return  np.exp(-(Si - Sj)**2/2) /np.sqrt(2*np.pi)

def Hijx2(Si, Sj):
    return -np.exp(-0.5 * (Si - Sj)**2) * (-3 + Si**2 - 6 * Si * Sj + Sj**2)/4/np.sqrt(2 * np.pi)


def Hijx4_x2(Si, Sj):
    exp_term = np.exp(-0.5 * (Si - Sj)**2)
    poly_term = (7 + Si**4 + 4 * Si**3 * Sj - 6 * Sj**2 + Sj**4 + 
                 6 * Si**2 * (-1 + Sj**2) + 4 * Si * Sj * (5 + Sj**2))
    denominator = 16 * np.sqrt(2 * np.pi)
    return exp_term * poly_term / denominator

def gauss_base(x,S):
    return np.sqrt(1/np.pi)*np.exp(-(x-S)**2)

def Hmat(vl,sl,Hij):
    n=len(vl)
    Hm=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            si=sl[i]
            sj=sl[j]
            Hm[i,j]=Hij(si,sj)
    return Hm
def Smat(vl,sl,Sij):
    n=len(vl)
    Sm=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            si=sl[i]
            sj=sl[j]
            Sm[i,j]=Sij(si,sj)
    return Sm


def wavefunc(x,sl,eigenvectors):
    y=np.zeros(x.shape)
    for i in range(len(eigenvectors)):
        y+=eigenvectors[i]*gauss_base(x,sl[i])
    return y
if __name__ == '__main__':

    sl=np.linspace(-20,20,100)
    vl=np.ones(sl.shape)
    H=Hmat(vl,sl,Hijx2)
    S=Smat(vl,sl,Sij)
    E, C= sp.linalg.eigh(H,S)
    fig=plt.figure(figsize=(14,6))
    lg=[]
    plt.subplot(1,2,1)
    x=np.linspace(-8,8,100000)
    for i in range(3):
        y=wavefunc(x,sl,C[:,i])
        plt.plot(x,y)
        lg.append(f"n={i}")
    plt.legend(lg)
    plt.xlabel("x",size=15)
    plt.ylabel(r"$\Psi(x)$",size=15)
    plt.subplot(1,2,2)
    plt.plot(x,x**2)
    plt.xlabel("x",size=15)
    plt.ylabel(r"$V(x)$",size=15)
    print("The three lowest energy of V(x)=x^2 is(h_bar=m=1):")
    print(E[:3])
    fig.savefig("wavefunction of V=x^2.png",dpi=600)

    H=Hmat(vl,sl,Hijx4_x2)
    S=Smat(vl,sl,Sij)
    E, C= sp.linalg.eigh(H,S)
    print("The three lowest energy of V(x)=x^4-x^2 is(h_bar=m=1):")
    print(E[:3])

    fig=plt.figure(figsize=(14,6))
    lg=[]
    plt.subplot(1,2,1)
    x=np.linspace(-8,8,100000)
    for i in range(3):
        y=wavefunc(x,sl,C[:,i])
        plt.plot(x,y)
        lg.append(f"n={i}")
    plt.legend(lg)
    plt.xlabel("x",size=15)
    plt.ylabel(r"$\Psi(x)$",size=15)
    plt.subplot(1,2,2)
    plt.plot(x,x**4-x**2)
    plt.xlabel("x",size=15)
    plt.ylabel(r"$V(x)$",size=15)
    fig.savefig("wavefunction of V=x^4-x^2.png",dpi=600)

