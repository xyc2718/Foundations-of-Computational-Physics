import numpy as np
import matplotlib.pyplot as plt
import os

"""
This is a simple pendulum class
:param m: mass of the pendulum
:param g: gravity
:param theta: angle of the pendulum
:param dtheta: angular velocity of the pendulum
:method geta: get the acceleration of the pendulum
:method getz: get the state of the pendulum
:method getHz: get the derivative of the state of the pendulum
:method update: update the state of the pendulum.It will automatically adjust the angle to be in the range of [-pi,pi]
"""
class Pendulum:
    def __init__(self,m=1,g=1,theta=0,dtheta=0) -> None:
        self.m=m
        self.g=g
        self.theta=theta
        self.dtheta=dtheta
    def geta(self):
        return -self.g*self.m*np.sin(self.theta)
    def getz(self):
        return np.array([self.theta,self.dtheta])
    def getHz(self):
        return np.array([self.dtheta,self.geta()])
    def update(self,theta,dtheta):
        self.theta=np.mod(theta+np.pi,2*np.pi)-np.pi
        self.dtheta=dtheta

def EulerStep(pendulum,dt):
    z=pendulum.getz()
    Hz=pendulum.getHz()
    z=z+Hz*dt
    pendulum.update(*z)

"""
Euler method for solving the pendulum
"""
def Euler(pendulum,dt,tmax):
    t=0
    zl=[]
    if dt<=0:
        raise ValueError("dt must be positive")
    while t<tmax:
        EulerStep(pendulum,dt)
        t+=dt
        zl.append(pendulum.getz())
    return np.array(zl)

def midpointStep(pendulum,dt):
    z=pendulum.getz()
    Hz=pendulum.getHz()
    z=z+Hz*dt/2
    pendulum.update(*z)
    Hz=pendulum.getHz()
    z=z+Hz*dt/2
    pendulum.update(*z)

"""
midpoint method for solving the pendulum
"""
def midpoint(pendulum,dt,tmax):
    t=0
    zl=[]
    if dt<=0:
        raise ValueError("dt must be positive")
    while t<tmax:
        midpointStep(pendulum,dt)
        t+=dt
        zl.append(pendulum.getz())
    return np.array(zl)
        
def Rk4step(pendulum,dt):
    z=pendulum.getz()
    Hz=pendulum.getHz()
    k1=Hz*dt
    pendulum.update(*(z+k1/2))
    Hz=pendulum.getHz()
    k2=Hz*dt
    pendulum.update(*(z+k2/2))
    Hz=pendulum.getHz()
    k3=Hz*dt
    pendulum.update(*(z+k3))
    Hz=pendulum.getHz()
    k4=Hz*dt
    pendulum.update(*(z+(k1+2*k2+2*k3+k4)/6))

"""
Runge-Kutta 4th order method for solving the pendulum
"""
def Rk4(pendulum,dt,tmax):
    t=0
    zl=[]
    if dt<=0:
        raise ValueError("dt must be positive")
    while t<tmax:
        Rk4step(pendulum,dt)
        t+=dt
        zl.append(pendulum.getz())
    return np.array(zl)

def EulerTrapezoidalStep(pendulum,dt,tol=1e-6,maxiter=1000):
    z0=pendulum.getz()
    Hz0=pendulum.getHz()
    z=z0.copy()
    for i in range(maxiter):
        Hzi=pendulum.getHz()
        pendulum.update(*(z0+Hz0*dt/2+Hzi*dt/2))
        zhis=z.copy()
        z=pendulum.getz()
        if np.linalg.norm(z-zhis)<tol:
            break
    else:
        print(f"waring: EulerTrapezoidalStep did not converge in {maxiter} iterations with tolerance {tol}")

"""
Euler trapezoidal method for solving the pendulum
"""
def EulerTrapezoidal(pendulum,dt,tmax,tol=1e-6,maxiter=100):
    t=0
    zl=[]
    if dt<=0:
        raise ValueError("dt must be positive")
    while t<tmax:
        EulerTrapezoidalStep(pendulum,dt,tol,maxiter)
        t+=dt
        zl.append(pendulum.getz())
    return np.array(zl)

"""
Test different intergrate method for the pendulum,will save the result to outputfold
:param i: index of the method 1-4 for Euler,midpoint,Rk4,EulerTrapezoidal
:param dt: time step
:param tmax: maximum time
:param theta0: initial angle
:param dtheta0: initial angular velocity
:param outputfold: output folder
"""
def test_intergrator(i,dt,tmax,theta0=1.0,dtheta0=-1.0,outputfold="output"):
    method=[Euler,midpoint,Rk4,EulerTrapezoidal]
    if dt<=0:
        raise ValueError("dt must be positive")
    if not os.path.exists(outputfold):
        os.makedirs(outputfold)
    pendulum=Pendulum(theta=theta0,dtheta=dtheta0)
    intergrator=method[i]
    intergratorname=intergrator.__name__
    zl=intergrator(pendulum,dt,tmax)
    tl=np.arange(zl.shape[0])*dt
    fig=plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    plt.plot(tl,zl[:,0])
    plt.xlabel("t/s")
    plt.ylabel(r"$\theta/rad$")
    plt.title(r"$\theta -t$"+f",dt={dt}s,method:{intergratorname}")
    plt.subplot(1,3,2)
    plt.plot(zl[:,0],zl[:,1])
    plt.xlabel(r"$\theta/rad$")
    plt.ylabel(r"$\dot{\theta}/(rad/s)$")
    plt.title(r"$\dot{\theta}-\theta$"+f",dt={dt}s,method:{intergratorname}")
    plt.subplot(1,3,3)
    m=pendulum.m
    g=pendulum.g
    plt.plot(tl,1/2*m*(zl[:,1])**2+m*g*(1-np.cos(zl[:,0])))
    plt.xlabel("t/s")
    plt.ylabel(r"$E/J$")
    plt.title(r"$E-t$"+f",dt={dt}s,method:{intergratorname}")
    fig.savefig(outputfold+f"\\pendulum_dt={dt}_method={intergratorname}.png",dpi=1000)
    print(f"save to {outputfold}\\pendulum_dt={dt}_method={intergratorname}.png")


if __name__=="__main__":
    for i in range(0,4):
        test_intergrator(i,0.1,100)
    for i in range(2,4):
        test_intergrator(i,1.0,1000)