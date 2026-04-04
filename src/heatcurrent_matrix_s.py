import numpy as np
# G1流动方向与z的正方向相反，G2流动方向与z的正方向相反

def Resistance(G1,G2,KA,z):
    """
    两相有源对流换热热阻
    Args:
        G1: 第一组流体的热容量流(J/K/s) scalar
        G2: 第二组流体的热容量流(J/K/s) scalar
        KA: 对流换热系数(W/m/K) scalar
        z: 积分段长度(m) scalar
    
    Returns:
        R: 热阻(K/W) scalar
    """
    # 热容量流G(J/K/s)，对流换热系数KA(W/m/K)，积分段长度z(m)
    if np.allclose(KA,0):
        R = float('inf')
    else:
        a2 = KA*(G2 + G1) / (G1*G2)
        R = (G1 + G2) / (G1*G2*(1-np.exp(-a2*z)))

    return R

def Phi1(G1,G2,KA,z,Q1,Q2):
    # 热容量流G(J/K/s)，对流换热系数KA(W/m/K)，积分段长度z(m)，热汇热源强度Q(W/m)
    if np.allclose(KA,0):
        s = Q1*z/G1
    else:
        a1 = (G2*Q1 - G1*Q2) / (KA*(G2 + G1))
        a2 = KA*(G2 + G1) / (G1*G2)
        phi1 = G1*G2*(np.exp(-a2*z)-1)*a1/(G1+G2) - G1*(Q1+Q2)*z/(G1+G2)
        s = phi1/G1
    return s

def Phi2(G1,G2,KA,z,Q1,Q2):
    # 热容量流G(J/K/s)，对流换热系数KA(W/m/K)，积分段长度z(m)，热汇热源强度Q(W/m)
    if np.allclose(KA,0):
        s = Q2*z/G2
    else:
        a1 = (G2*Q1 - G1*Q2) / (KA*(G2 + G1))
        a2 = KA*(G2 + G1) / (G1*G2)
        phi2 = G1*G2*(1-np.exp(-a2*z))*a1/(G1+G2) - G2*(Q1+Q2)*z/(G1+G2)
        s = phi2/G2
    return s

def setAa_s(N, zlist, KA, G1, G2, T1in, T2in, Q1list, Q2list):
    # input:分段数N，各段高度zlist，1相和2相在各分段的容量流G1和G2，1相进口值T1in，2相进口值T2in，
    #       各分段的源Q1和源Q2
    # print('setA')
    A11 = np.eye(N+1)
    A31 = np.zeros([N,N+1])
    A32 = np.zeros([N,N+1]) 
    for i in range(N):
        A11[i,i+1] = -1
        A31[i,i+1] = -1
        A32[i,i+1] = 1
    A22 = A11.copy()
    A12 = np.zeros([N+1,N+1])
    A21 = np.zeros([N+1,N+1])
    A13 = np.zeros([N+1,N])
    A23 = np.zeros([N+1,N])
    A33 = np.zeros([N,N])
    a1 = np.zeros(N+1)
    a1[N] = T1in
    a2 = np.zeros(N+1)
    a2[N] = T2in
    aa = np.zeros(N)
    for i in range(N):
        A13[i,i] = 1/G1[i]
        A23[i,i] = -1/G2[i]
        A33[i,i] = Resistance(G1[i],G2[i],KA[i],zlist[i])
        a1[i] = Phi1(G1[i],G2[i],KA[i],zlist[i],Q1list[i],Q2list[i])
        a2[i] = Phi2(G1[i],G2[i],KA[i],zlist[i],Q1list[i],Q2list[i])  
    # 拼接矩阵
    A1 = np.append(A11,A21,axis=0)
    A1 = np.append(A1,A31,axis=0)
    A2 = np.append(A12,A22,axis=0)
    A2 = np.append(A2,A32,axis=0)
    A3 = np.append(A13,A23,axis=0)
    A3 = np.append(A3,A33,axis=0)
    A = np.append(A1,A2,axis=1)
    A = np.append(A,A3,axis=1)  # 参数矩阵A
    aa = np.append(a2,aa,axis=0)
    aa = np.append(a1,aa,axis=0)
    return A,aa

