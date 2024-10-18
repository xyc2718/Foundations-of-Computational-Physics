import numpy as np

"""
partial-pivoting scheme of Matrix M
"""
def partial_pivoting_scheme(M: np.array, i: int) -> np.array:
    reM=M.copy()
    max_index = i + np.argmax(np.abs(M[i:, i]))
    if i != max_index:
        # Swap the rows
        reM[max_index,:] = M[i, :]
        reM[i, :] = M[max_index, :]
    return reM

"""
Gaussian elimination of Matrix M and solve x
:param M: input matrix
:return: the matrix after Gaussian elimination
:return: the solution x
:return: flag=0, there is a unique solution of Matrix
        flag=1, there are Infinitely many solutions of Matrix, the indeterminate x will be set to 1
        flag=2, there is No solution of Matrix, nan x is return
"""
def gaussian_elimination(M:np.array) -> list[np.array, np.array,int]:
    Me=M.copy()
    flag=0
    # Forward elimination
    row=Me.shape[0]
    for i in range(row):
        Me=partial_pivoting_scheme(Me,i)
        for j in range(i+1,row):
            Me[j,:]=Me[j,:]-Me[i,:]*Me[j,i]/Me[i,i]
    # Backward substitution
    x=np.zeros(row)

    for i in range(row-1,-1,-1):
        for j in range(row-1,i,-1):
            x[i]-=Me[i,j]*x[j]

        x[i]+=Me[i,-1]
        if Me[i,i]==0 and x[i]!=0:
            flag=2
            return Me,np.nan,flag
        if Me[i,i]==0 and x[i]==0:
            x[i]=1
            flag=1
            continue
        x[i]/=Me[i,i]
        

    return Me,x,flag

if __name__ == '__main__':
    M=np.loadtxt('MatrixIn.txt',delimiter=",",dtype=float)
    Me,x,flag=gaussian_elimination(M)
    print("Input matrix is:")
    print(M)
    print("The matrix after Gaussian elimination is:")
    print(Me)
    if flag==0:
        print("The solution x is:")
        print(x)
    if flag==1:
        print("There are Infinitely many solutions of Matrix, the indeterminate x will be set to 1,one of solution is:")
        print(x)
    if flag==2:
        print("There is No solution of Matrix")