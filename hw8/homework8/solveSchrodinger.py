import numpy as np
import matplotlib.pyplot as plt
import argparse
class SchrodingerSolver:
    def __init__(self,Lx,Nx,V,dt,ifwarningoverflow=True):
        self.V=V
        self.Lx=Lx
        self.Nx=Nx
        self.init_grid()
        self.dx=self.xl[1]-self.xl[0]
        self.dt=dt
        self.phimax=10e8
        self.ifwarningoverflow=ifwarningoverflow
    def init_grid(self):
        self.xl=np.linspace(-self.Lx,self.Lx,self.Nx)
        self.u=np.zeros_like(self.xl,dtype=complex)
        self.apply_boundary_(self.u)
    def apply_boundary_(self,u):
        u[0]=u[-1]=0.0+0.0j

    def set_initial_condition(self,u0):
        self.u=u0(self.xl)
        self.apply_boundary_(self.u)

    def update_u_(self,u):
        self.u=u
        self.apply_boundary_(self.u)
    def init_crank_nicolson_mat(self,theta=0.5):
        """
        initialize the matrix for Crank-Nicolson scheme
        """
        ap=self.dt/(self.dx**2)
        B=np.zeros((self.Nx,self.Nx))
        V=np.zeros((self.Nx,self.Nx))
        I=np.eye(self.Nx)
        for i in range(0,self.Nx):
            B[i,i]=2
            V[i,i]=self.V(self.xl[i])*self.dt
            if i<self.Nx-1:
                B[i,i+1]=-1
                B[i+1,i]=-1
        self.Bmat=B
        self.Vmat=V
        A=np.linalg.inv(I*2j-2*theta*V-theta*ap*B)@(I*2j+2*(1-theta)*V+(1-theta)*ap*B)
        self.Acn=A
    def crank_nicolson(self,maxstep):
        """
        solve the Schrodinger equation by Crank-Nicolson scheme.this should be called after init_crank_nicolson_mat and set_initial_condition
        """
        ul=[]
        for i in range(maxstep):
            phi2=np.abs(self.u)**2
            if np.any(phi2>self.phimax):
                if self.ifwarningoverflow:
                    print(f"Crank_Nicolson Method :Iteration stopped at step {i} due to overflow or invalid values.")
                return np.array(ul)
            ul.append(self.u)
            self.update_u_(self.Acn@self.u)
        return np.array(ul)
    def stable_explicit(self,maxstep):
        """
        solve the Schrodinger equation by stable explicit scheme.The first step is calculated by Crank-Nicolson scheme.
        """
        ap=self.dt/(self.dx**2)
        self.init_crank_nicolson_mat(theta=0.5)
        ul=[self.u.copy()]
        self.update_u_(self.Acn@self.u)
        for i in range(maxstep):
            phi2=np.abs(self.u)**2
            if np.any(phi2>self.phimax):
                if self.ifwarningoverflow:
                    print(f"Stable Explicit Method:Iteration stopped at step {i} due to overflow or invalid values.")
                return np.array(ul)
            ul.append(self.u.copy())
            self.update_u_(ul[-2]-1j*ap*self.Bmat@self.u-2j*self.Vmat@self.u)
        return np.array(ul)


def square_well(x,a=2.0,V0=-1.0):
    if (x>=-a and x<=a):
        return V0
    else:
        return 0
    
def gaussian(x,x0=-5.0,k0=1.0):
    return np.exp(-((x-x0)**2)/2)*np.exp(1j*k0*x)

def main(Lx=20,Nx=100,dt=0.01,maxstep=1000):
        print(f"Solving Schrodinger equation with square well potential with Lx={Lx},Nx={Nx},dt={dt},maxstep={maxstep}")
        zth=1e6
        ss=SchrodingerSolver(Lx,Nx,square_well,dt=dt)
        ss.set_initial_condition(lambda x:gaussian(x)-gaussian(Lx))
        ss.init_crank_nicolson_mat(theta=0.5)
        ul1=ss.crank_nicolson(maxstep)
        tl1=np.arange(0,ul1.shape[0])*dt
        T1,X1=np.meshgrid(tl1,ss.xl)
        Un1=np.transpose(np.abs(ul1)**2)

        ss.set_initial_condition(lambda x:gaussian(x)-gaussian(Lx))
        ss.init_crank_nicolson_mat(theta=0.0)
        ul2=ss.crank_nicolson(maxstep)
        tl2=np.arange(0,ul2.shape[0])*dt
        T2,X2=np.meshgrid(tl2,ss.xl)
        Un2=np.transpose(np.abs(ul2)**2)

        ss.set_initial_condition(lambda x:gaussian(x)-gaussian(Lx))
        ul3=ss.stable_explicit(maxstep)
        tl3=np.arange(0,ul3.shape[0])*dt
        Un3=np.transpose(np.abs(ul3)**2)
        T3,X3=np.meshgrid(tl3,ss.xl)
        fig=plt.figure()
        fig=plt.figure(figsize=(16,5))
        ax1=fig.add_subplot(131,projection='3d')
        ax1.plot_surface(T1,X1,Un1,cmap='rainbow',alpha=0.7)
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        zmin,zmax=ax1.get_zlim()
        ax1.set_zlim(zmin,min(zmax,zth))
        ax1.set_zlabel('probiability density')
        ax1.set_title(f"Solve by Crank-Nicolson scheme with \n dx={ss.dx:.3f},dt={ss.dt:.3f},alpha={ss.dt/ss.dx**2:.4f}")
        ax1.view_init(elev=30, azim=-30)
        ax2 = fig.add_subplot(133, projection='3d')
        ax2.plot_surface(T2, X2, Un2, cmap='rainbow', alpha=0.7)
        ax2.set_xlabel('t')
        ax2.set_ylabel('x')
        ax2.set_title(f'Solve by explicit scheme with \n dx={ss.dx:.3f},dt={ss.dt:.3f},alpha={ss.dt/ss.dx**2:.4f}')
        ax2.view_init(elev=30, azim=-30)
        zmin, zmax = ax2.get_zlim()
        ax2.set_zlim(max(zmin,-zth), min(zmax,zth))
        ax3 = fig.add_subplot(132, projection='3d')
        ax3.plot_surface(T3, X3, Un3, cmap='rainbow', alpha=0.7)
        ax3.set_xlabel('t')
        ax3.set_ylabel('x')
        ax3.set_title(f'Solve by stable explicit scheme \n with dx={ss.dx:.3f},dt={ss.dt:.3f},alpha={ss.dt/ss.dx**2:.4f}')
        ax3.view_init(elev=30, azim=-30)

        fig.savefig(f'solveSchrodinger_Nx={Nx}_dt={dt:.3f}.png',dpi=800)
        print(f"Save figure to solvebyCrankNicolson_Nx={Nx}_dt={dt:.3f}.png")

def find_Valpha(Lx=20.0,dt=0.01,Vr=110.0,apb=0.1,ape=1.2,x0=5.0,k0=-1.0,maxstep=50,umax=10000.0,NV=40,Na=40,testmethod='stable_explicit',theta=0.5):
    apl=[]
    Vl=[]
    for Vi in np.linspace(-Vr,Vr,NV):
        def testwell(x):
            return square_well(x,V0=Vi)
        for api in np.linspace(apb,ape,Na):
            Ni=np.int64(round(np.sqrt(4*Lx**2*api/dt)))

            st=SchrodingerSolver(Lx,Ni,testwell,dt=dt,ifwarningoverflow=False)
            st.set_initial_condition(lambda x:gaussian(x,x0=x0,k0=k0))
            if testmethod=='crank_nicolson':
                st.init_crank_nicolson_mat(theta=theta)
                ut=st.crank_nicolson(maxstep)
            elif testmethod=='stable_explicit':
                ut=st.stable_explicit(maxstep)
            else:
                raise ValueError("Invalid testmethod")
            if np.max(np.abs(ut[-1])**2)>umax:
                 apl.append(st.dt/(st.dx**2))
                 Vl.append(Vi*dt)
        # print(Vi)
    return np.array(apl),np.array(Vl)

def test_convergence_stable_explicit(Lx=20.0,dt=0.01,Vr=120.0,apb=0.01,ape=1.2,x0=5.0,k0=-1.0,maxstep=50,umax=100.0,NV=40,Na=40):
    print(f"\nTest convergence of stable explicit method with parameters:Lx={Lx},dt={dt},Vr={Vr},apb={apb},ape={ape},x0={x0},k0={k0},maxstep={maxstep},umax={umax},NV={NV},Na={Na}\nThis will take a few minutes\n")
    apl,Vl=find_Valpha(Lx,dt,Vr,apb,ape,x0,k0,maxstep,umax,NV,Na,testmethod='stable_explicit')
    Ne=np.sqrt(4*Lx**2*ape/dt)
    fig=plt.figure()
    plt.scatter(Vl,apl,marker="x",c='b')
    plt.xlim(-Vr*dt-10*dt,Vr*dt+10*dt)
    plt.ylim(-dt*20**2/Lx,dt/(2*Lx/(Ne+10))**2)
    plt.ylabel(r'$\alpha$')
    plt.xlabel(r"$V_0 \Delta t$")
    plt.title("Convergence Test for Stable Explicit Method")
    x1=np.linspace(-Vr*dt-10*dt,Vr*dt+10*dt,100)
    y1=np.ones_like(x1)*0.5
    y2=-0.5*x1+0.5
    y3=np.zeros_like(x1)
    y0=np.linspace(-dt*20**2/Lx,dt/(2*Lx/(2*Ne+10))**2,100)
    x0=np.ones_like(y0)*(-1.0)
    plt.plot(x1,y1,c='r',linestyle='--')
    plt.legend(["Not Converge Points","Stability boundary from Von Neumann stability analysis."])
    plt.plot(x1,y2,c='r',linestyle='--')
    plt.plot(x1,y3,c='r',linestyle='--')
    plt.plot(x0,y0,c='r',linestyle='--')
    fig.savefig("ConvergenceTestStableExplicit.png",dpi=800)
    
def test_convergence_crank_nicolson(Lx=20.0,dt=0.01,Vr=150.0,apb=0.01,ape=1.2,x0=5.0,k0=-1.0,maxstep=50,umax=100.0,NV=40,Na=40,theta=0.5):
    print(f"\nTest convergence of Crank-Nicolson method with parameters:theta={theta},Lx={Lx},dt={dt},Vr={Vr},apb={apb},ape={ape},x0={x0},k0={k0},maxstep={maxstep},umax={umax},NV={NV},Na={Na}\nThis will take a few minutes\n")
    apl,Vl=find_Valpha(Lx,dt,Vr,apb,ape,x0,k0,maxstep,umax,NV,Na,testmethod='crank_nicolson',theta=theta)
    # fig=plt.figure()
    Ne=np.sqrt(4*Lx**2*ape/dt)
    plt.scatter(Vl,apl,marker="x",c='b')
    plt.xlim(-Vr*dt-10*dt,Vr*dt+10*dt)
    plt.ylim(-dt*20**2/Lx,dt/(2*Lx/(Ne+10))**2)
    plt.ylabel(r'$\alpha$')
    plt.xlabel(r"$V_0 \Delta t$")
    plt.title(f"Convergence Test for Crank Nicolson Method at theta={theta}")

    plt.legend(["Not Converge Points"])
    # fig.savefig(f"ConvergenceTestCrankNicolson_theta={theta}.png",dpi=800)


if __name__=='__main__':
    main(Lx=20.0,Nx=100,dt=0.01,maxstep=1000)
    main(Lx=20.0,Nx=112,dt=0.01,maxstep=1000)
    main(Lx=20.0,Nx=283,dt=0.01,maxstep=1000)
    main(Lx=20.0,Nx=284,dt=0.01,maxstep=1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_convergence",action="append",choices=["SE","CN"],default=[],help="Test convergence of stable explicit method(SE) or Crank-Nicolson method(CN)")
    args=parser.parse_args()
    if "SE" in args.test_convergence:
        test_convergence_stable_explicit()
    if "CN" in args.test_convergence:
        fig=plt.figure(figsize=(12,10))
        plt.subplot(2,2,1)
        test_convergence_crank_nicolson(theta=0.0)
        plt.subplot(2,2,2)
        test_convergence_crank_nicolson(theta=0.1)
        plt.subplot(2,2,3)
        test_convergence_crank_nicolson(theta=0.3)
        plt.subplot(2,2,4)
        test_convergence_crank_nicolson(theta=0.5)
        fig.savefig("ConvergenceTestCrankNicolson.png",dpi=1000)



