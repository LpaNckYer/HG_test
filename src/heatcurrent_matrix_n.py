import numpy as np
# G1流动方向与z的正方向一致，G2流动方向与z的正方向相反

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

    if np.allclose(G1,G2):
        if np.allclose(KA,0):
            R = float('inf')
        else:
            a = KA*z / G1
            R = (a + 1) / (a*G1)
    else:   # 两相流体热容量流不相等
        if np.allclose(KA,0):
            R = float('inf')
        else:
            a2 = KA*(G2 - G1) / (G1*G2)
            R = ((G2-G1)**2 + G1*G2*(np.exp(-a2*z)-1)*(np.exp(a2*z)-1)) / (G1*G2*(np.exp(-a2*z)-1)*(G1*np.exp(a2*z)-G2))        
    return R

def Resistance_n_pc(G2,KA,z):
    # 相变段热阻（G1侧相变）
    # 热容量流G(J/K/s)，对流换热系数KA(W/m/K)，积分段长度z(m)
    a2 = - KA / G2

    R = 1 / ((np.exp(-a2*z) - 1)*G2)

    return R

def Phi1(G1,G2,KA,z,Q1,Q2):
    """
    Args:
        G1: 第一组流体的热容量流(J/K/s) scalar
        G2: 第二组流体的热容量流(J/K/s) scalar
        KA: 对流换热系数(W/m/K) scalar
        z: 积分段长度(m) scalar
        Q1: 第一组热源的热源强度(W/m) scalar
        Q2: 第二组热源的热源强度(W/m) scalar
    
    Returns:
        s: 附加源(K) scalar
    """
    # 热容量流G(J/K/s)，对流换热系数KA(W/m/K)，积分段长度z(m)，热汇热源强度Q(W)
    if np.allclose(KA,0):
        s = Q1*z/G1
    else:
        if np.allclose(G1,G2):  # 两相流体热容量流相等
            a = KA*z / G1
            phi1 = z/2 * ((a+2)/(a+1)*Q1 - a/(a+1)*Q2)
            s = phi1/G1
        else:   # 两相流体热容量流不相等
            q1 = Q1
            q2 = Q2
            a1 = (G2*q1 - G1*q2) / (KA*(G2 - G1))
            a2 = KA*(G2 - G1) / (G1*G2)

            phi1 = ((G1*G2*(np.exp(-a2*z)-1)*(G1*np.exp(a2*z)-G2))*a1 + G1*G2*(np.exp(-a2*z)-1)*(q2-q1)*z + G1*(G2-G1)*(q2-q1)*z) / ((G2-G1)**2 + G1*G2*(np.exp(-a2*z)-1)*(np.exp(a2*z)-1))
            
            s = phi1/G1
    return s

def Phi1_pc():
    # 相变段附加源（G1侧相变）
    return 0.0

def Phi2(G1,G2,KA,z,Q1,Q2):
    # 热容量流G(J/K/s)，对流换热系数KA(W/m/K)，积分段长度z(m)，热汇热源强度Q(W/m)
    if np.allclose(KA,0):
        s = Q2*z/G1
    else:
        if np.allclose(G1,G2):  # 两相流体热容量流相等
            a = KA*z / G1
            phi2 = z/2 * (a/(a+1)*Q1 - (a+2)/(a+1)*Q2)
            s = phi2/G2
        else:   # 两相流体热容量流不相等
            q1 = Q1
            q2 = Q2
            a1 = (G2*q1 - G1*q2) / (KA*(G2 - G1))
            a2 = KA*(G2 - G1) / (G1*G2)

            phi2 = (-(G1*G2*(np.exp(-a2*z)-1)*(G1*np.exp(a2*z)-G2))*a1 + G1*G2*(np.exp(a2*z)-1)*(q2-q1)*z - G2*(G2-G1)*(q2-q1)*z) / ((G2-G1)**2 + G1*G2*(np.exp(-a2*z)-1)*(np.exp(a2*z)-1))
            
            s = phi2/G2
    return s

def Phi2_pc(G2,KA,z,Q1,Q2):
    """
    Args:
        G2: 第二组流体的热容量流(J/K/s) scalar
        KA: 对流换热系数(W/m/K) scalar
        z: 积分段长度(m) scalar
        Q1: 第一组热源的热源强度(W/m) scalar
        Q2: 第二组热源的热源强度(W/m) scalar
    
    Returns:
        s: 附加源(K) scalar    
    """
    q1 = Q1
    q2 = Q2
    a1 = q2 / KA
    a2 = KA / G2

    phi2 = -(1-np.exp(-a2*z))*a1 - (q2-q1)*z/G2

    s = phi2/G2
    return s

def setAa_n(N, zlist, KA, G1, G2, T1in, T2in, Q1list, Q2list):
    # input:分段数N，各段高度zlist，1相和2相在各分段的容量流G1和G2，1相进口值T1in，2相进口值T2in，
    #       各分段的源Q1和源Q2
    # print('setA')
    A11 = np.eye(N+1)
    A31 = np.zeros([N,N+1])
    A32 = np.zeros([N,N+1]) 
    for i in range(N):
        A11[i+1,i] = -1
        A31[i,i] = -1
        A32[i,i+1] = 1
    A22 = np.transpose(A11)
    A12 = np.zeros([N+1,N+1])
    A21 = np.zeros([N+1,N+1])
    A13 = np.zeros([N+1,N])
    A23 = np.zeros([N+1,N])
    A33 = np.zeros([N,N])
    a1 = np.zeros(N+1)
    a1[0] = T1in
    a2 = np.zeros(N+1)
    a2[N] = T2in
    aa = np.zeros(N)
    for i in range(N):
        A13[i+1,i] = 1/G1[i]
        A23[i,i] = -1/G2[i]
        A33[i,i] = Resistance(G1[i],G2[i],KA[i],zlist[i])
        a1[i+1] = Phi1(G1[i],G2[i],KA[i],zlist[i],Q1list[i],Q2list[i])
        a2[i] = Phi2(G1[i],G2[i],KA[i],zlist[i],Q1list[i],Q2list[i])  # 气相源项
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

def setAa_n_pc(N, zlist, KA, G2, T1in, T2in, Q1list, Q2list):
    """G1相变段
    """
    # input:分段数N，各段高度zlist，2相在各分段的容量流G2，1相进口值T1in，2相进口值T2in，
    #       各分段的源Q1和源Q2
    # print('setA')
    A11 = np.eye(N+1)
    A31 = np.zeros([N,N+1])
    A32 = np.zeros([N,N+1]) 
    for i in range(N):
        A11[i+1,i] = -1
        A31[i,i] = -1
        A32[i,i+1] = 1
    A22 = np.transpose(A11)
    A12 = np.zeros([N+1,N+1])
    A21 = np.zeros([N+1,N+1])
    A13 = np.zeros([N+1,N])
    A23 = np.zeros([N+1,N])
    A33 = np.zeros([N,N])
    a1 = np.zeros(N+1)
    a1[0] = T1in
    a2 = np.zeros(N+1)
    a2[N] = T2in
    aa = np.zeros(N)
    for i in range(N):
        A13[i+1,i] = 0
        A23[i,i] = -1/G2[i]
        A33[i,i] = Resistance_n_pc(G2[i],KA[i],zlist[i])
        a1[i+1] = Phi1_pc()
        a2[i] = Phi2_pc(G2[i],KA[i],zlist[i],Q1list[i],Q2list[i])  # 气相源项
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