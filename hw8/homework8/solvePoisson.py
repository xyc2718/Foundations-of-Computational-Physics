import numpy as np
import matplotlib.pyplot as plt

class PoissonSolver:
    def __init__(self,Lx,Ly,Nx,Ny,bu,bd,bl,br,rho):
        self.upbound=bu
        self.downbound=bd
        self.leftbound=bl
        self.rightbound=br
        self.rho=rho
        self.Lx=Lx
        self.Ly=Ly
        self.Nx=Nx
        self.Ny=Ny
        self.dx=Lx/Nx
        self.dy=Ly/Ny
        self.ap=self.dx**2/self.dy**2
        self.init_grid()
        self.rhol=rho(self.X,self.Y)

    def init_grid(self):
        self.xl=np.linspace(0,self.Lx,self.Nx)
        self.yl=np.linspace(0,self.Ly,self.Ny)
        self.u=np.zeros((self.Nx,self.Ny))
        self.X,self.Y=np.meshgrid(self.xl,self.yl)
        self.apply_boundary_(self.u)

    def boundary_pos(self):
        """
        return the position of the boundary
        """
        bu=np.zeros((self.Nx,self.Ny),dtype=bool)
        bd=np.zeros((self.Nx,self.Ny),dtype=bool)
        bl=np.zeros((self.Nx,self.Ny),dtype=bool)
        br=np.zeros((self.Nx,self.Ny),dtype=bool)
        bu[-1,:]=1
        bd[1,:]=1
        bl[:,1]=1
        br[:,-1]=1
        return bu,bd,bl,br

    def apply_boundary_(self,u):
        """
        apply boundary condition to the solution u
        """
        bu,bd,bl,br=self.boundary_pos()
        u[bu]=self.upbound(self.X[bu])
        u[bd]=self.downbound(self.X[bd])
        u[bl]=self.leftbound(self.Y[bl])
        u[br]=self.rightbound(self.Y[br])
    def update_u_(self,u):
        self.u=u
        self.apply_boundary_(self.u)

    def jacobi_step_(self):
        """
        perform one step of Jacobi iteration
        """
        u=np.zeros_like(self.u)
        u[1:-1,1:-1]=(self.u[:-2,1:-1]+self.u[2:,1:-1]+self.ap*(self.u[1:-1,:-2]+self.u[1:-1,2:])+self.dx**2*self.rhol[1:-1,1:-1])/2/(1+self.ap)
        self.update_u_(u)
        return self.u
    def jacobi_(self,maxiter=10000,tol=1e-1):
        """
        solve the Poisson equation using Jacobi iteration
        """
        hisig=self.u.copy()
        for i in range(maxiter):
            self.jacobi_step_()
            if np.max(np.abs(self.u-hisig))<tol*self.dx**2:
                print(f"Jacobi converged after {i} iterations with relative tolerance {tol}")
                break
            hisig=self.u.copy()
        else:
            print(f"Jacobi did not converge after {maxiter} iterations with relative tolerance {tol}")
        return self.u
    

if __name__=="__main__":
    Lx=1
    Ly=1.5
    Nx=100
    Ny=100
    print(f"Solving Poisson equation with Lx={Lx},Ly={Ly},Nx={Nx},Ny={Ny}")
    def upbound(x):
        return np.zeros_like(x)
    def downbound(x):
        return np.zeros_like(x)
    def leftbound(y):
        return np.zeros_like(y)
    def rightbound(y):
        return np.zeros_like(y)
    def rho(x,y):
        return np.ones_like(x)
    ps1=PoissonSolver(1,1,100,100,upbound,downbound,leftbound,rightbound,rho)
    u1=ps1.jacobi_()

    def upbound(x):
        return np.ones_like(x)
    def downbound(x):
        return np.zeros_like(x)
    def leftbound(y):
        return np.zeros_like(y)
    def rightbound(y):
        return np.zeros_like(y)
    def rho(x,y):
        return np.zeros_like(x)
    
    Lx=1
    Ly=1.5
    Nx=100
    Ny=100
    print(f"Solving Poisson equation with Lx={Lx},Ly={Ly},Nx={Nx},Ny={Ny}")
    xl=np.linspace(0,Lx,Nx)
    yl=np.linspace(0,Ly,Ny)
    ps2=PoissonSolver(Lx,Ly,Nx,Ny,upbound,downbound,leftbound,rightbound,rho)
    u2=ps2.jacobi_()
    fig=plt.figure(figsize=(10,5))
    ax1=fig.add_subplot(121,projection='3d')
    ax1.plot_surface(ps1.X,ps1.Y,u1,cmap='rainbow')
    ax1.set_xlabel("X/m")
    ax1.set_ylabel("Y/m")
    ax1.set_zlabel("u/V")
    ax1.set_title("Poisson equation with condition 1")

    ax2=fig.add_subplot(122,projection='3d')
    ax2.plot_surface(ps2.X,ps2.Y,u2,cmap='rainbow')
    ax2.set_xlabel("X/m")
    ax2.set_ylabel("Y/m")
    ax2.set_zlabel("u/V")
    ax2.set_title("Poisson equation with condition 2")

    fig.savefig("solvePoisson.png",dpi=800)
