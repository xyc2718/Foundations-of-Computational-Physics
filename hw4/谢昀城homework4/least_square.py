import numpy as np
import matplotlib.pyplot as plt

"""
using SVD to solve least square problem
"""
def least_square(A,b):
    U, S, VT = np.linalg.svd(A.T, full_matrices=False)
    x=VT.T@np.linalg.inv(np.diag(S))@U.T@b.T
    return x

if __name__ == '__main__':
    xl = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    xl=xl.reshape(1,-1)
    Tl = np.array([14.6, 18.5, 36.6, 30.8, 59.2, 60.1, 62.2, 79.4, 99.9])
    xl1=np.vstack((np.ones(xl.shape),xl))
    xl2=np.vstack((np.ones(xl.shape),xl,xl**2))
    a0,a1=least_square(xl1,Tl)
    a0,a1,a2=least_square(xl2,Tl)
    x=np.linspace(np.min(xl),np.max(xl),100)
    fig=plt.figure()
    plt.plot(x,a0+a1*x)
    plt.plot(x,a0+a1*x+a2*x**2)
    plt.scatter(xl,Tl,c="r",marker="s")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.legend(["Linear","Quadratic","Data"])
    fig.savefig("least_square.png")
    r1=np.sum((a0+a1*xl-Tl)**2)
    r12=1-np.sum((a0+a1*xl-Tl)**2)/np.sum((Tl-np.mean(Tl))**2)
    r2=np.sum((a0+a1*xl+a2*xl**2-Tl)**2)
    r22=1-np.sum((a0+a1*xl+a2*xl**2-Tl)**2)/np.sum((Tl-np.mean(Tl))**2)
    print("straight-line fit:")
    print(f"T={a0:.3f}+{a1:.3f}x")
    print(f"least square={r1:.3f}")
    print(f"R^2={r12:.1f}")
    print("quadratic-line fit:")
    print(f"T={a0:.3f}+{a1:.3f}x+{a2:.3f}x^2")
    print(f"least square={r2:.3f}")
    print(f"R^2={r22:.2f}")
