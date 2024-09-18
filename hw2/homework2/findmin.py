import numpy as np
import matplotlib.pyplot as plt
def g(x,y):  # function to be minimized and it's gradient
    return np.sin(x+y)+np.cos(x+2*y)
def dg(x,y):
    return np.array([np.cos(x+y)-np.sin(x+2*y),np.cos(x+y)-2*np.sin(x+2*y)])

def gradiant_descent(X0,g,dg,ap=0.01,tol=1e-8,max_iter=100000):
    """
    gradiant descent algorithm to find the minimum of a function
    x,y: initial guess
    ap: learning rate
    tol: tolerance
    max_iter: maximum number of iterations
    """
    X=np.array(X0,dtype=float)
    his=[]
    convenged=0
    for i in range(max_iter):
        dX=dg(*X)
        X-=ap*dX+np.random.normal(loc=0.0, scale=tol)  # add noise to avoid vanishing gradient at local minimum
        his.append(X.copy())
        if np.linalg.norm(dX)<tol:
            print(f"gradiant_descent converged after {i} iterations")
            break
    else:
        convenged=1
    if convenged:
        print(f"gradiant_descent did not convenged after {max_iter} iterations")
    return X,np.array(his)
if __name__=="__main__":
    Xmin,his=gradiant_descent([0,0],g,dg,tol=1e-6,max_iter=100000)

    ##output
    print(f"minimum of g(x) is {g(*Xmin):.5f}")
    print(f"Xmin is {Xmin[0]:.5f}+ 2 m pi,Ymin is {Xmin[1]:.5f}+2 n pi, m,n are any integers")

    ##plot
    fig=plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    x=np.linspace(-8,8,100)
    y=np.linspace(-5,8,100)
    X,Y=np.meshgrid(x,y)
    Z=g(X,Y)
    heatmap = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.plot(his[:,0],his[:,1],c="b",linewidth=3);
    plt.plot(x,-x,c="r")
    for n in range(4):  #plot the cell
        plt.plot(x,(n-1)*2*np.pi-x,c="r")
        plt.plot(x,(n-1)*np.pi-x/2,c="r")
    plt.plot(x,-x/2,c="r")   
    plt.xlim(-8,8)
    plt.ylim(-5,8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("(a)heatmap of g(x) and gradiant descent path")
    plt.colorbar(heatmap)
    plt.legend(["path of gradiant descent","cells of g(x,y)"])
    plt.subplot(1,2,2)
    plt.plot([g(his[i,0],his[i,1]) for i in range(len(his))])
    plt.xlabel("iteration")
    plt.ylabel("function value")
    plt.title("(b)function value - iteration")
    fig.savefig("find g(x,y) min.png",dpi=600)
    plt.show()