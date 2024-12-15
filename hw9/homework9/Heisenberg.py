import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def marsaglia():
    while True:
        v1 = 2*np.random.rand()-1
        v2 = 2*np.random.rand()-1
        s = v1**2 + v2**2
        s2=np.sqrt(1-s)
        if s<1:
            return np.array([2*v1*s2,2*v2*s2,1-2*s])
class Heisenberg:
    def __init__(self,L,T,J=-1.0):
        self.L = L
        self.T = T
        self.J = J
        self.baiss = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        self.generate_fcc_lattice()
        self.set_spins()
        self.initialize_neighbors()
    def generate_fcc_lattice(self):
        lattice = []
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    for b in self.baiss:
                        lattice.append([i, j, k] + b)
        self.lattice=np.array(lattice)
    def set_spins(self,init_spin=[]):
        if init_spin == []:
            self.spins = np.random.normal(size=(len(self.lattice), 3))
        else:
            self.spins = np.array([init_spin for _ in range(len(self.lattice))],dtype=float)
    def initialize_neighbors(self):
        neighbors = []
        nbdistance=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        tol=1e-5
        for i, ri in enumerate(self.lattice):
            neighbor = []
            for j, rj in enumerate(self.lattice):
                rij=ri-rj
                self.apply_PBC_(rij)
                rij=np.abs(rij)
                if i != j and np.any(np.all(np.isclose(nbdistance, rij, atol=tol), axis=1)):
                    # print(rij)
                    neighbor.append(j)
            neighbors.append(neighbor)
            neighborsarr=np.array(neighbors)
        self.neighbors=neighborsarr
    def apply_PBC_(self,rij):
        for i in range(3):
            if rij[i]>self.L/2:
                rij[i]-=self.L
            elif rij[i]<-self.L/2:
                rij[i]+=self.L
    def calculate_energy(self):
        sn=self.spins[self.neighbors]
        return self.J*np.sum(self.spins[:, np.newaxis, :] * sn)/2/self.lattice.shape[0]
    def calculate_magnetization(self):
        return np.sum(self.spins,axis=0)/self.lattice.shape[0]
    
    def calculate_dE(self,i,spini1,spini2):
        ds=spini2-spini1
        dE=self.J*np.sum(ds*self.spins[self.neighbors[i]])
        return dE
    
    def metropolis_step(self):
        index = np.random.randint(len(self.spins))
        s1 = self.spins[index]
        s2 = marsaglia()
        dE=self.calculate_dE(index,s1,s2)
        if dE<0 or np.random.rand()<np.exp(-dE/self.T):
            self.spins[index]=s2
    def visualize(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        lattice=self.lattice
        spins=self.spins
        ax.quiver(lattice[:, 0], lattice[:, 1], lattice[:, 2],
                spins[:, 0], spins[:, 1], spins[:, 2], length=0.3, normalize=True)
        ax.set_title(f"3D Visualization of Spins (T={self.T},J={self.J},L={self.L},N={self.lattice.shape[0]})")
        return fig
    
def findTc(Llist=[2,3,4]):
    col=len(Llist)
    Tl=np.linspace(0.4,8,50)
    Nl=np.zeros((col,Tl.shape[0]))
    Eave=np.zeros((col,Tl.shape[0]))
    Mave=np.zeros((col,Tl.shape[0]))
    Estd=np.zeros((col,Tl.shape[0]))
    Mstd=np.zeros((col,Tl.shape[0]))
    for i,L in enumerate(Llist):
        eqstep=1000*L**3
        spstep=2000*L**3
        sf=1*L**3
        Nl[i,:]=L**3*4
        for j,T in enumerate(Tl):
            hs=Heisenberg(L,T)
            hs.set_spins([1,0,0])
            for _ in range(eqstep):
                hs.metropolis_step()
            El=[]
            Ml=[]
            print(f"calculate at T={T},L={hs.L},J={-hs.J},sites={hs.lattice.shape[0]}")
            for k in range(spstep):
                hs.metropolis_step()
                if k%sf==0:
                    E=hs.calculate_energy()
                    M=hs.calculate_magnetization()
                    El.append(E)
                    Ml.append(np.linalg.norm(M))
            Eave[i,j]=np.mean(El)
            Mave[i,j]=np.mean(Ml)
            Estd[i,j]=np.std(El)
            Mstd[i,j]=np.std(Ml)
    Tc=Tl[np.argmax(((Mstd**2)/Tl)[-1,:])]
    fig=plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    y=Eave.T
    plt.plot(Tl,y,marker="s")
    plt.plot([Tc,Tc],[np.min(y),np.max(y)],linestyle="--")
    lg=([f"L={L}" for L in Llist])
    lg.append("Tc")
    plt.legend(lg)
    plt.xlabel(r"$k_B T/J$")
    plt.ylabel("<E>")
    plt.subplot(2,2,2)
    y=Mave.T
    plt.plot(Tl,y,marker="s")
    plt.plot([Tc,Tc],[np.min(y),np.max(y)],linestyle="--")
    plt.xlabel(r"$k_B T/J$")
    plt.ylabel("|<M>|")
    lg=([f"L={L}" for L in Llist])
    lg.append("Tc")
    plt.legend(lg)
    plt.subplot(2,2,3)
    y=((Estd**2)/Tl/Tl).T
    y=y*(Nl.T)
    plt.plot(Tl,y,marker="s")
    plt.plot([Tc,Tc],[np.min(y),np.max(y)],linestyle="--")
    plt.xlabel(r"$k_B T/J$")
    plt.ylabel("C")
    lg=([f"L={L}" for L in Llist])
    lg.append("Tc")
    plt.legend(lg)
    plt.subplot(2,2,4)
    y=((Mstd**2)/Tl).T
    y=y*(Nl.T)
    plt.plot(Tl,y,marker="s")
    plt.plot([Tc,Tc],[np.min(y),np.max(y)],linestyle="--")
    plt.xlabel(r"$k_B T/J$")
    plt.ylabel(r"$\chi$")
    lg=([f"L={L}" for L in Llist])
    lg.append("Tc")
    plt.legend(lg)
    fig.savefig("findTc.png",dpi=600)
    print(f"\n#########\nTc={Tc}\n##########")

    

    
if __name__=="__main__":
    mars=np.array([marsaglia() for _ in range(10000)])
    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    ax.scatter(mars[:,0],mars[:,1],mars[:,2],s=1)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_title("Marsaglia Sampling")
    fig.savefig("Marsaglia.png",dpi=600)

    print("calculate L=4,T=1.0")
    hs=Heisenberg(4,1.0)
    hs.set_spins([1,0,0])
    for _ in range(100000):
        hs.metropolis_step()
    fig=hs.visualize()
    fig.savefig("T=1.0.png",dpi=600)
    print("calculate L=4,T=6.0")
    hs=Heisenberg(4,6.0)
    hs.set_spins([1,0,0])
    for _ in range(100000):
        hs.metropolis_step()
    fig=hs.visualize()
    fig.savefig("T=6.0.png",dpi=600)

    findTc([2,3,4,5])


