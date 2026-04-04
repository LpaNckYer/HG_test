import numpy as np

def smooth_heaviside(x, k=10):
    """平滑的阶跃函数，k控制过渡速度"""
    return 1 / safe_1_plus_exp(x, k)

def piecewise_smooth(x, x0, func1, func2, k=10):
    """分段函数平滑过渡：x < x0 时 func1，x >= x0 时 func2"""
    weight = smooth_heaviside(x - x0, k)
    return (1 - weight) * func1(x) + weight * func2(x)

# # 示例：在 x=2 处从线性函数过渡到二次函数
# x = np.linspace(0, 4, 100)
# y = piecewise_smooth(x, x0=2, 
#                     func1=lambda x: x, 
#                     func2=lambda x: x**2,
#                     k=10)  # k越大过渡越陡峭


def safe_1_plus_exp(x, k=1):
    """
    安全计算 1 + exp(-k*x)，避免溢出
    """
    exponent = -k * x
    
    # 当 exponent 很大正数时，1 + exp(exponent) ≈ exp(exponent)
    # 当 exponent 很大负数时，1 + exp(exponent) ≈ 1
    # 当 exponent 适中时，直接计算
    
    # 设置阈值，避免溢出
    threshold = 700  # 对于float64，exp(709)接近最大可表示值
    result = np.zeros_like(exponent)
    mask1 = exponent > threshold
    mask2 = exponent < -threshold
    result[mask1] = 1.0 + np.exp(threshold)
    result[mask2] = 1.0
    result[(~mask1) & (~mask2)] = 1.0 + np.exp(exponent[(~mask1) & (~mask2)])

    return result