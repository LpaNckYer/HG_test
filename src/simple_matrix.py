import numpy as np

def setAa_linear_n(N, z_list, T_in, a_list, b_list):
    """
    为逆流简单变量设置矩阵A和向量a
    形如：dT/dz = a*T + b
    Args:
        N: 分段数
        zlist：各段高度     (N,)
        T_in：进口值
        a_list：各段a系数   (N,)
        b_list：各段b系数   (N,)
    Returns:
        A: 系数矩阵         (N+1,N+1)
        a: 向量             (N+1,)
    """
    A = np.zeros((N+1,N+1))
    a = np.zeros(N+1)
    zero_indices = np.where(a_list==0)
    # print(zero_indices)
    for i in range(N):
        A[i,i] = - a_list[i] * np.exp(a_list[i]*z_list[i])
        A[i,i+1] = a_list[i]
        a[i] = b_list[i] * np.exp(a_list[i]*z_list[i]) - b_list[i]
    for i in zero_indices:
        A[i,i] = -1
        A[i,i+1] = 1
        a[i] = b_list[i]*z_list[i]
    A[N][N] = 1
    a[N] = T_in

    return A, a

def setAa_p(N, z_list, p2_in, a_list):
    """
    为顺流压力简单变量设置矩阵A和向量a
    形如：dp/dz = a/p
    Args:
        N: 分段数
        zlist：各段高度     (N,)
        p2_in：进口值的平方
        a_list：各段a系数   (N,)
    Returns:
        A: 系数矩阵         (N+1,N+1)
        a: 向量             (N+1,)
    """
    A = np.zeros((N+1,N+1))
    a = np.zeros(N+1)
    for i in range(N):
        A[i][i] = 1
        A[i+1][i] = -1
        a[i+1] = 2 * a_list[i] * z_list[i]
    A[N][N] = 1
    a[0] = p2_in    
    return A, a

def setAa_constant_s(N, z_list, T_in, a_list):
    """
    为顺流简单变量设置矩阵A和向量a
    形如：dT/dz = a
    Args:
        N: 分段数
        zlist：各段高度     (N,)
        T_in：进口值
        a_list：各段a系数   (N,)
    Returns:
        A: 系数矩阵         (N+1,N+1)
        a: 向量             (N+1,)
    """
    A = np.zeros((N+1,N+1))
    a = np.zeros(N+1)
    for i in range(N):
        A[i+1][i] = - 1
        A[i+1][i+1] = 1
        a[i+1] = a_list[i]*z_list[i]
    A[0][0] = 1
    a[0] = T_in
    return A, a

def setAa_constant_n(N, z_list, T_in, a_list):
    """
    为逆流简单变量设置矩阵A和向量a
    形如：dT/dz = a
    Args:
        N: 分段数
        zlist：各段高度     (N,)
        T_in：进口值
        a_list：各段a系数   (N,)
    Returns:
        A: 系数矩阵         (N+1,N+1)   
        a: 向量             (N+1,)
    """
    A = np.eye(N+1)
    a = np.zeros(N+1)
    for i in range(N):
        A[i][i+1] = - 1
        a[i] = a_list[i]*z_list[i]
    A[N][N] = 1
    a[N] = T_in
    return A, a