import numpy as np
from scipy.special import erf
import scipy as sp
import matplotlib.pyplot as plt
E0=27.211386
rmax=50
r = np.linspace(0.01, rmax, 1000)  # Avoid r=0 to prevent division by zero
Z_ion = 3.0
r_loc = 0.4
C1, C2, C3, C4 = -14.0093922, 9.5099073, -1.7532723, 0.0834586

def VLi(r, Z_ion=Z_ion, r_loc=r_loc, C1=C1, C2=C2, C3=C3, C4=C4):
    term1 = -Z_ion / r * erf(r / np.sqrt(2)/r_loc)
    term2 = np.exp(-0.5 * (r / r_loc)**2)
    polynomial = (C1 +
                  C2 * (r / r_loc)**2 +
                  C3 * (r / r_loc)**4 +
                  C4 * (r / r_loc)**6)
    V = term1 + term2 * polynomial
    return V
def VH(r):
    return -1/r

"""
effective V
"""
def Veff(r,l,type="Li"):
    if type=="Li":
        return VLi(r)+l*(l+1)/(2*r**2)
    if type=="H":
        return VH(r)+l*(l+1)/(2*r**2)

"""
Solve Schrodinger equation by uniform grid
"""
def solve_schrodinger(l,n=1000,rmin=0, rmax=100,type="Li"):
    H=np.zeros((n,n))
    dr=(rmax-rmin)/n
    r=np.linspace(dr,rmax,n,endpoint=False)
    for i in range(n):
        H[i,i]=1+Veff(r[i],l,type=type)*dr**2
        if i<n-1:
            H[i,i+1]=-1/2
            H[i+1,i]=-1/2
    re=np.linalg.eigh(H)
    El=re[0]/dr**2
    # sorted_indices = sorted(range(len(El)), key=lambda i: El[i])
    sorted_indices=np.argsort(El)
    El = El[sorted_indices]
    vl=(re[1])[:,sorted_indices]
    return El,r,vl


def int_Trapezoidal(fl,rl):
    dr=rl[:1]-rl[:-1]
    intfx=sum([(fl[i+1]-fl[i])*dr[i] for i in range(len(fl)-1)])/2
    return intfx

"""
plot wavefunction normalized
"""
def plot_wavefunction(r,vl,label=""):
    A=np.sqrt(int_Trapezoidal(vl**2,r))
    plt.plot(r,(vl/A),label=label)

"""
Solve Schrodinger equation by ununiform grid
"""
def solve_schrodinger_nonuniform(l,jmax=1000,rmin=0, rmax=100,delta=0.01,type="Li"):
    H=np.zeros((jmax-1,jmax-1))
    rp=rmax/(np.exp(delta*jmax)-1)
    def a(j):
        return (2*rp**2*delta**2*np.exp(2*j*delta))
    for j in range(1,jmax):
        rj=rp*(np.exp(delta*j)-1)
        H[j-1,j-1]=(2+delta**2/4)/a(j)+Veff(rj,l,type=type)
        if j<jmax-1:
            H[j-1,j]=-1/a(j)
            H[j,j-1]=-1/a(j+1)
    re=np.linalg.eig(H)
    El=re[0]
    sorted_indices=np.argsort(El)
    El = El[sorted_indices]
    s=((np.exp(delta*np.arange(1,jmax)/2)).reshape(-1, 1))
    vl=(re[1])[:,sorted_indices]*s
    r=np.array([rp*(np.exp(delta*j)-1) for j in range(1,jmax)])
    
    return El,r,vl

"""
print results
"""
def printresult(Earray,nm):
    flat=Earray.ravel()
    indices=np.argsort(flat)[:nm]
    maxpositions=np.unravel_index(indices, Earray.shape)
    maxpositions = np.array(list(zip(maxpositions[0], maxpositions[1])))
    for i in range(nm):
        l,n=maxpositions[i]
        print(f"n={n+1}, l={l}, E={Earray[l,n]*E0}eV")

if __name__=="__main__":
    ##plot potential
    potentialLi = VLi(r)
    potentialH = VH(r)
    fig=plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(r, potentialH)
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.title("Potential of H")
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(r, potentialLi)
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.title("Potential of Li")
    plt.grid()
    fig.savefig("potentialLi.png",dpi=600)
    
    ###calulate by uniform grid

    maxl=4
    ngrid=1000
    print(f"\n############\nsolve schrodinger of Li and H by uniform grid:n={ngrid},rmax={rmax}")
    Earray=np.ones((maxl,maxl))*100
    varray = np.empty((maxl, maxl,ngrid), dtype=float)
    for l in range(maxl):
        El,r,vl = solve_schrodinger(l,n=ngrid,rmax=rmax,type="H")
        Earray[l,l:]=El[:maxl-l]
        varray[l,l:]=np.transpose(vl[:,:maxl-l])
    print(f"\nfirst 3 energy of H:")
    printresult(Earray,6)

    ##solve schrodinger of Li

    Earray=np.ones((maxl,maxl))*100
    varray = np.empty((maxl, maxl,ngrid), dtype=float)
    for l in range(maxl):
        El,r,vl = solve_schrodinger(l,n=ngrid,rmax=rmax,type="Li")
        Earray[l,l:]=El[:maxl-l]
        varray[l,l:]=np.transpose(vl[:,:maxl-l])
    print(f"\nfirst 3 energy of Li:")
    printresult(Earray,3)

    ###calulate by ununiform grid 
    maxl=4
    jmax=1000
    delta=5/jmax
    print(f"\n############\nsolve schrodinger of Li and H by ununiform grid:jmax={jmax},delta={delta:.3f},rmax={rmax}\n")
    ##solve schrodinger of H
    Earray=np.ones((maxl,maxl))*100
    varray = np.empty((maxl, maxl,jmax-1), dtype=float)
    rarray = np.empty((maxl,maxl, jmax-1), dtype=float)
    for l in range(maxl):
        El,r,vl = solve_schrodinger_nonuniform(l,jmax=jmax,delta=delta,rmax=rmax,type="H")
        Earray[l,l:]=El[:maxl-l]
        varray[l,l:]=np.transpose(vl[:,:maxl-l])
        rarray[l,l:]=r
    
    fig=plt.figure(figsize=(12,8))

    for n in range(maxl):
        plt.subplot(2,2,n+1)
        for l in range(n+1):
            plot_wavefunction(rarray[l,n],varray[l,n],label=f"n={n+1}, l={l}, E={Earray[l,n]*E0:.4f}eV")
        plt.xlabel("r")
        plt.ylabel("u(r)")
        # if n>0:
        #     plt.ylim(min(varray[0,n]/2),max(varray[0,n]/2))
        plt.title(f"Wavefunctions of n={n+1},H")
        plt.grid()
        plt.legend()
    fig.savefig("wavefunctionsH.png",dpi=1000)

    print("first 3 energy of H:")
    printresult(Earray,6)

    ##solve schrodinger of Li
    Earray=np.ones((maxl,maxl))*100
    varray = np.empty((maxl, maxl,jmax-1), dtype=float)
    rarray = np.empty((maxl,maxl, jmax-1), dtype=float)
    for l in range(maxl):
        El,r,vl = solve_schrodinger_nonuniform(l,jmax=jmax,delta=delta,rmax=rmax,type="Li")
        Earray[l,l:]=El[:maxl-l]
        varray[l,l:]=np.transpose(vl[:,:maxl-l])
        rarray[l,l:]=r

    fig=plt.figure(figsize=(12,8))
    for n in range(maxl):
        plt.subplot(2,2,n+1)
        for l in range(n+1):
            plot_wavefunction(rarray[l,n],varray[l,n],label=f"n={n+1}, l={l}, E={Earray[l,n]*E0:.4f}eV")
        plt.xlabel("r")
        plt.ylabel("u(r)")
        # if n>0:
        #     plt.ylim(min(varray[0,n]/2),max(varray[0,n]/2))
        plt.title(f"Wavefunctions of n={n+1},H")
        plt.grid()
        plt.legend()
    fig.savefig("wavefunctionsLi.png",dpi=1000)

    print(f"\nfirst 3 energy of Li:")
    printresult(Earray,3)


    ##plot grid spacing
    fig=plt.figure()
    plt.plot(rarray[0,0][1:]-rarray[0,0][:-1])
    plt.xlabel("r")
    plt.ylabel(r"$\Delta r$")
    plt.title("ununiform grid spacing")
    fig.savefig("ununiformgrid.png",dpi=600)


    ##compare uniform and nonuniform grid
fig=plt.figure(figsize=(12,5))
El1=[]
El2=[]
tn=2
tl=0
minc=50
maxc=500
dm=10
for k in range(minc,maxc,dm):
    E1= (solve_schrodinger(tl,n=k,rmax=rmax,type="H"))[0]
    E2=(solve_schrodinger_nonuniform(tl,jmax=k,delta=6/k,rmax=rmax,type="H"))[0]
    El1.append(E0*E1[tn-tl-1]+E0/2/tn**2)
    El2.append(E0*E2[tn-tl-1]+E0/2/tn**2)
plt.subplot(1,2,1)
plt.plot( range(minc,maxc,dm),El1,label="uniform grid")
plt.xlabel("n")
plt.ylabel("E-E0")
plt.title(f"uniform grid of H, n={tn},l={tl}")
plt.subplot(1,2,2)
plt.plot( range(minc,maxc,dm),El2,label="ununiform grid")
plt.xlabel("n")
plt.ylabel("E-E0")
plt.title(f"ununiform grid of H, n={tn},l={tl}")
fig.savefig("compare.png",dpi=600)


    
