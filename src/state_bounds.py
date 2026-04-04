"""
BVP / 热量流子程序共用的状态变量物理限域（单点维护，避免各 ODE 内重复 clip）。"""
import numpy as np

# 温度类状态 [K]（气相 T、固相 t 共用同一 envelope）
BVP_TEMP_MIN = 200.0
BVP_TEMP_MAX = 2500.0

# 摩尔分数 / 还原度等 [0, 1]
BVP_FRACTION_MIN = 0.0
BVP_FRACTION_MAX = 1.0

# 压力 [Kg/m2]
BVP_P_MIN = 1e3
BVP_P_MAX = 3e4

# 床层密度 [kg/m3 bed]
BVP_RHOB_MIN = 500.0
BVP_RHOB_MAX = 3000.0


def clip_state_up8(T, t, fs, x, y, w, rho_b, p):
    """上半部 BVP 状态 (T,t,fs,x,y,w,rhob,p)，与 blast_furnace_bvp 列次序一致。"""
    return (
        np.clip(T, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(t, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(fs, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(x, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(y, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(w, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(rho_b, BVP_RHOB_MIN, BVP_RHOB_MAX),
        np.clip(p, BVP_P_MIN, BVP_P_MAX),
    )


def clip_state_down6(T, t, fs, x, rho_b, p):
    """下半部 BVP 状态 (T,t,fs,x,rhob,p)。"""
    return (
        np.clip(T, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(t, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(fs, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(x, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(rho_b, BVP_RHOB_MIN, BVP_RHOB_MAX),
        np.clip(p, BVP_P_MIN, BVP_P_MAX),
    )


def clip_thermal_moles_pressure_7(T, t, fs, x, y, w, p):
    """热量流：无 rhob 时的 T,t,fs,x,y,w,p。"""
    return (
        np.clip(T, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(t, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(fs, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(x, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(y, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(w, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(p, BVP_P_MIN, BVP_P_MAX),
    )


def clip_down_core_5(T, t, fs, x, p):
    """下半部热量流常用：T,t,fs,x,p（无 rhob）。"""
    return (
        np.clip(T, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(t, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(fs, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(x, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(p, BVP_P_MIN, BVP_P_MAX),
    )


def clip_T_x_p(T, x, p):
    """下半部 p 方程等：T, x, p。"""
    return (
        np.clip(T, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(x, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(p, BVP_P_MIN, BVP_P_MAX),
    )


def clip_down_gas_p_inputs(T, fs, x, p):
    """下半部 p_hc / dpdz：T, fs, x, p。"""
    return (
        np.clip(T, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(fs, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(x, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(p, BVP_P_MIN, BVP_P_MAX),
    )


def clip_gas_dp_inputs(T, x, y, w, p):
    """上半部 dpdz / p_hc：T,x,y,w,p。"""
    return (
        np.clip(T, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(x, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(y, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(w, BVP_FRACTION_MIN, BVP_FRACTION_MAX),
        np.clip(p, BVP_P_MIN, BVP_P_MAX),
    )


def clip_iter_temperatures(T, t):
    """热量流内迭代中对 T、t 的再约束。"""
    return (
        np.clip(T, BVP_TEMP_MIN, BVP_TEMP_MAX),
        np.clip(t, BVP_TEMP_MIN, BVP_TEMP_MAX),
    )
