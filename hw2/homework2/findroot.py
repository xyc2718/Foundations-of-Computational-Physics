import numpy as np
import matplotlib.pyplot as plt
import decimal
import os
decimal.getcontext().prec = 30

def f(x):
    return x**3-5*x+3

def df(x):
    return 3*x**2-5

def f_decimal(x):  # Define a function with decimal type to handle high precision numbers
    xd = decimal.Decimal(x)
    return xd**3 - 5*xd + 3

def df_decimal(x):
    xd = decimal.Decimal(x)
    return 3*xd**2 - 5

def find_bracket(f, rg=[-3, 3], n=10, ifplot=False):
    """
    Find the bracket of the root of the function f in the range rg by sampling n points
    :param f: function
    :param rg: range
    :param n: number of points
    :return: the bracket of the root
    """
    x = np.linspace(rg[0], rg[1], n)
    y = f(x)
    y_roll = np.roll(y, 1)
    y_cov = y[1:] * y_roll[1:]
    cr0 = np.where(y_cov < 0)  # Convolve 2 series to find the cross point
    bracket = [(x[i], x[i+1]) for i in cr0[0]]
    if ifplot:  # Plot the function and the bracket if ifplot is True
        for i in range(len(bracket)):
            xx = np.linspace(rg[0], rg[1], 1000)
            plt.plot(xx, f(xx))
            plt.plot(bracket[i], [0, 0], c='r')
            plt.scatter(bracket[i], [0, 0], c='b', marker="s")
    return bracket

def bisection_method(f, rg, tol=1e-4, max_iter=100):
    """
    Bisection method to find the root of the function f
    :param f: function
    :param rg: range    
    :param tol: tolerance
    :param max_iter: maximum iteration
    """
    a, b = rg
    if f(a) * f(b) >= 0:
        raise ValueError("Function does not change sign in the interval.")
    converged = 0
    for i in range(max_iter):
        c = (a + b) / 2
        if np.abs(b - a) < tol:  # Convergence condition, using |a-b| as the error
            prc = int(-np.log10(abs(b - a)))
            print(f"The bisection method converged after {i} iterations")
            print(f"The root is {c:.{prc+1}f}", f"The error is {np.abs(b - a):.{1}e}")  # Formatted output
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    else:
        converged = 1
    if converged:
        print("The bisection method did not converge after", max_iter, "iterations.")
    return (a + b) / 2, np.abs(b - a)

def newton_raphson_method(f_decimal, df_decimal, x0, tol, max_iter=100):
    """
    Newton-Raphson method to find the root of the function f
    :param f_decimal: function with decimal type
    :param df_decimal: derivative of f_decimal
    :param x0: initial point
    :param tol: tolerance
    :param max_iter: maximum iteration
    """
    x = decimal.Decimal(x0)
    tol = decimal.Decimal(tol)  # Convert parameters to decimal type
    converged = 0
    for i in range(max_iter):
        fx = f_decimal(x)
        dfx = df_decimal(x)
        if dfx == 0:
            raise ValueError("Meet zero derivative at ", x)
        dx = fx / dfx
        if abs(dx) < tol:  # Use dx=fx/dfx as the error
            prc = int(-decimal.Decimal.log10(abs(dx)))
            print(f"The Newton-Raphson method converged after {i} iterations")
            print(f"The root is {x:.{prc+1}f}\n", f"The error is {abs(dx):.{1}e}")  # Formatted output
            break
        x = x - dx
    else:
        converged = 1
        
    if converged:
        print("The Newton-Raphson method did not converge after", max_iter, "iterations")

    return x, abs(dx)

def hybrid_method(f, df, rg, tol=1e-15, max_iter=100):
    """
    Hybrid method combines Newton-Raphson method and bisection method
    :param f: function
    :param df: derivative of f
    :param rg: range
    :param tol: tolerance
    :param max_iter: maximum iteration
    """
    a = decimal.Decimal(rg[0])  # Convert parameters to decimal type
    b = decimal.Decimal(rg[1])
    eps = 10**(5 - (decimal.getcontext()).prec)
    x = (a + b) / 2
    converged = 0
    if f(a) * f(b) >= 0:
        raise ValueError("Function does not change sign in the interval.")
    for i in range(max_iter):
        dfx = df(x)
        fx = f(x)
        if abs(dfx) < eps:  # If dfx is less than eps, use the bisection method
            x = (a + b) / 2
            dx = abs(b - a)
        else:
            dx = fx / dfx
            x = x - dx
        if abs(dx) < tol:
            prc = int(-decimal.Decimal.log10(abs(dx)))
            print(f"The hybrid method converged after {i} iterations")
            print(f"The root is {x:.{prc+1}f}\n", f"The error is {abs(dx):.{1}e}")
            break
    else:
        converged = 1
    if converged:
        print("The hybrid method did not converge after", max_iter, "iterations.")
    return x, abs(dx)

fig = plt.figure()
bracket = find_bracket(f, [-3, 3], n=10, ifplot=True) #get bracket and plot
bracket = bracket[1::] # Only take the positive interval
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) with bracket")
plt.legend(["f(x)", "bracket"])
fig.savefig("f(x) with bracket.png", dpi=600) #save figure

nsol = len(bracket)

if __name__ == "__main__":
    # Call functions to solve respectively
    print("\n#########################\n")
    bisection_root = []
    for i in range(0, nsol):
        root, err = bisection_method(f, bracket[i], tol=1e-5, max_iter=100)
        bisection_root.append(root)

    print("\n#########################\n")
    for i in range(0, nsol):
        newton_raphson_method(f_decimal, df_decimal, bisection_root[i], tol=1e-15, max_iter=100)

    print("\n#########################\n")
    for i in range(0, nsol):
        hybrid_method(f_decimal, df_decimal, bracket[i], tol=1e-15, max_iter=100)
    plt.show()
    os.system("pause")

