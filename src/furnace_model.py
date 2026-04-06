import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm, cond
from scipy.integrate import solve_bvp

from sigmoid import smooth_heaviside
from heatcurrent_matrix_n import setAa_n
from heatcurrent_matrix_s import setAa_s
from simple_matrix import setAa_linear_n, setAa_p, setAa_constant_s, setAa_constant_n

from constant import pai, R, R_, g_c, T_std, P_std, eps
from state_bounds import (
    BVP_FRACTION_MIN,
    BVP_FRACTION_MAX,
    BVP_P_MAX,
    BVP_P_MIN,
    BVP_TEMP_MAX,
    BVP_TEMP_MIN,
    clip_furnace_state_up8,
    clip_iter_temperatures,
)
from hc_solver_settings import (
    HC_REL_TOL_MAIN,
    HC_REL_TOL_TIGHT,
    HC_MAX_ITER_TT_XY_FS,
    HC_MAX_ITER_W,
    HC_RELAXATION,
    HC_MAX_ITER_TEST_NESTED_INNER,
    HC_MAX_ITER_TEST_OUTER_UP,
    HC_MAX_ITER_TEST_FIRST_UP,
)


def _nonneg_sqrt(x):
    """避免 Re、分数等因浮点噪声略负时 x**(1/2) 触发 RuntimeWarning。"""
    return np.sqrt(np.maximum(np.asarray(x, dtype=float), 0.0))


def _nonneg_cbrt(x):
    """Pr、Sc 等非负相关量开 1/3 次；略负时截断为 0 再 cbrt。"""
    return np.cbrt(np.maximum(np.asarray(x, dtype=float), 0.0))


def _heme_shell_1_minus_fs(fs):
    """颗粒扩散项中的 (1-fs+eps)：fs 限到 ≤1、去 NaN/inf，再下限 eps，避免负底数或非有限值进入分数幂。"""
    fs_a = np.asarray(fs, dtype=float)
    fs_s = np.nan_to_num(fs_a, nan=0.0, posinf=1.0, neginf=0.0)
    fs_eff = np.minimum(fs_s, 1.0)
    raw = 1.0 - fs_eff + eps
    shell = np.maximum(raw, eps)
    return shell


def _solid_T_for_diffusion_powers(t):
    """扩散系数 t^1.78：去 NaN/inf 后 clip 到 [BVP_TEMP_MIN, BVP_TEMP_MAX]，避免非有限或越界温度进幂。"""
    t0 = np.asarray(t, dtype=float)
    raw = np.nan_to_num(
        t0,
        nan=float(BVP_TEMP_MIN),
        posinf=float(BVP_TEMP_MAX),
        neginf=float(BVP_TEMP_MIN),
    )
    out = np.clip(raw, float(BVP_TEMP_MIN), float(BVP_TEMP_MAX))
    return out


class FurnaceModel:
    """高炉计算模型"""

    def __init__(self, parameters):
        self.params = parameters
        self.results = {}
        self.last_bvp_profile_df = None
        self.last_hc_profile_df = None

    def run(self):
        """运行模型"""
        print(f"计算中：{self.params.case_name}")

        H0 = self.params.H0
        HH = self.params.HH

        y_guess, H_ctrl = self.params.initial_bvp_guess()
        
        # 求解
        final_sol, history = self.solve_with_decreasing_tol(
            self.blast_furnace_bvp, 
            self.bc, 
            H_ctrl, 
            y_guess,
            tol_levels=[1e-3]
        )
        
        # 输出结果
        print("\n=== 迭代历史 ===")
        for i, record in enumerate(history):
            print(f"轮次 {i+1}: 容差={record['tol']:.1e}, "
                f"节点数={record['n_nodes']}, 成功={record['success']}")
        
        # 绘制结果
        y_plot = final_sol.y
        x_plot = final_sol.x
        
        y_plot = final_sol.sol(x_plot)
        # plt.figure(figsize=(12, 8))
        variables = ['T', 't', 'fs', 'x', 'y', 'w', 'rhob', 'p']
        # for i in range(8):
        #     plt.subplot(3, 3, i+1)
        #     plt.plot(x_plot, y_plot[i])
        #     plt.ylabel(variables[i])
        #     plt.xlabel('z (m)')
        # plt.tight_layout()
        # plt.show()

        # 剖面 CSV 由网格无关性 / 初值范围测试脚本写出（见 scripts/test_grid_independence.py 等）
        df = pd.DataFrame(np.vstack((x_plot, y_plot)).T, columns=['z'] + variables)
        self.last_bvp_profile_df = df
        # df.to_csv(f'bvp_{H0:.1f}-{HH:.1f}m_loop.csv', index=False)

        last_h = history[-1]
        rr = getattr(final_sol, "rms_residuals", None)
        if rr is not None and np.size(rr) > 0:
            bvp_max_rms = float(np.max(rr))
        else:
            bvp_max_rms = None
        self.results = {
            "case_name": self.params.case_name,
            "H0": x_plot[0],
            "HH": x_plot[-1],
            "T_out": y_plot[0,0],
            "t_out": y_plot[1,-1],
            "fs_out": y_plot[2,-1],
            "x_out": y_plot[3,0],
            "y_out": y_plot[4,0],
            "w_out": y_plot[5,0],    
            "rhob_out": y_plot[6,-1],    
            "p_bottom": y_plot[7,-1],
            "bvp_success": bool(final_sol.success),
            "bvp_tol_final": float(last_h["tol"]),
            "bvp_n_nodes_final": int(last_h["n_nodes"]),
            "bvp_max_rms_residual_final": bvp_max_rms,
            "bvp_bc_l2_residual_final": None,
        }

        return self.results
    
    # solving
    def solve_with_decreasing_tol(self, ode, bc, x_span, y_init, tol_levels=None):
        """
        使用逐步减小容差的方法求解BVP
        
        参数:
        - ode: 微分方程函数
        - bc: 边界条件函数  
        - x_span: 求解区间
        - y_init: 初始猜测
        - tol_levels: 容差级别列表，默认[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        
        返回:
        - solution: 最终解
        - history: 各轮迭代结果历史
        """
        
        if tol_levels is None:
            tol_levels = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        
        # 初始网格
        # x = np.linspace(x_span[0], x_span[-1], self.params.initial_mesh)
        x = np.linspace(x_span[0], x_span[-1], len(y_init[0]))    # 改为使用初始猜测的节点数
        
        history = []
        
        for i, tol in enumerate(tol_levels):
            print(f"第 {i+1} 轮迭代，容差: {tol}")
            
            # 求解BVP
            sol = solve_bvp(ode, bc, x, y_init, tol=tol, max_nodes=len(x)*50, verbose=2)
            
            if not sol.success:
                print(f"警告: 第 {i+1} 轮迭代未收敛")
                # 即使未完全收敛，仍使用当前解作为下一轮初始值
                if i == 0:
                    # 第一轮就失败，可能需要调整初始猜测
                    raise RuntimeError("初始求解失败，请检查问题设置")
            
            # 记录结果
            history.append({
                'tol': tol,
                'solution': sol,
                'success': sol.success,
                'n_nodes': len(sol.x)
            })
            
            # 为下一轮准备：使用当前解作为初始猜测
            # 可以增加网格点数以提高精度
            # x = np.linspace(x_span[0], x_span[-1], min(self.params.initial_mesh, len(sol.x) * 2))
            x = np.linspace(x_span[0], x_span[-1], len(sol.x))    # 改为使用初始猜测的节点数
            y_init = sol.sol(x)
        
        return sol, history    

    # bvp definition
    def blast_furnace_bvp(self,Z,Y):
        """
        Args:
            Z: height. ndarray. (n,)
            Y: state variables (T,t,fs,x,y,w,rho_b,p). ndarray. (m,n)
        Returns:
            dY/dz: space derivative of state variables. ndarray. (m,n)
        """
        m, n = Y.shape
        res = np.empty((m, n))
        for i in range(n):
            z = Z[i]
            T, t, fs, x, y, w, rho_b, p = clip_furnace_state_up8(*Y[:, i])
            res[:,i] = [self.dTdz(z,T,t,fs,x,y,w,p),
                        self.dtdz(z,T,t,fs,x,y,w,p,rho_b),
                        self.dfsdz(z,T,t,fs,x,y,w,p),
                        self.dxdz(z,T,t,fs,x,y,w,p),
                        self.dydz(z,T,t,fs,x,y,w,p),
                        self.dwdz(z,T,t,fs,x,y,w,p),
                        self.drhobdz(z,T,t,fs,x,y,w,p),
                        self.dpdz(z,T,x,y,w,p)]
        return res

    def bc(self,ya,yb):
        """
        Args:
            ya: boundary condition of state variables at z=0. ndarray. (n,)
            yb: boundary condition of state variables at z=H. ndarray. (n,)
        Returns:
            bc: boundary condition. ndarray. (n,)
        """
        return np.array([yb[0]-self.params.T_in,
                        ya[1]-self.params.t_in,
                        ya[2]-self.params.fs_in,
                        yb[3]-self.params.x_in,
                        yb[4]-self.params.y_in,
                        yb[5]-self.params.w_in,
                        ya[6]-self.params.rhob_in,
                        ya[7]-self.params.p_in])

    # odes
    def dTdz(self,z,T,t,fs,x,y,w,p):
        """differential equation of T
        temperature of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]   
            U (float): overall heat transfer coefficient based on inner surface area of furnace-wall. [kcal / m2 * hr * K]
            T_we (float): exit tempareture of cooling water. [K]

        Returns:
            dd (float): [K / m]

        """
        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        C,dCdT = self.HeatCapacity_Gas(T,x,y,w) # C (float): heat capacity of gas. [kcal / kg * K] ; dCdT (float): differential of C with T. [kcal / kg * K**2]

        q1 = 0.0  # [kcal / m3 bed * hr] 热源项占位（未建模）
        q2 = self.Heat_2(z,T,t,fs,x,y,w,p) # [(kmol / m3 bed * hr) * (kg / m3)]
        q3 = self.Heat_3(z,T,t,x,y,w) # [kcal / m3 bed * hr]

        dd = (Az * (q1 + 22.4*C*q2*T + q3) + pai * Dz * self.params.U * (T - self.params.T_we)) / (rho * F * (C + T*dCdT))

        return dd
    
    def dtdz(self,z,T,t,fs,x,y,w,p,rho_b):
        """differential equation of t
        temperature of solid particle

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
            rho_b (float): bulk density of solid particles. [kg / m3 bed]
        Operate:    
            Fs (float): volume rate of solid particles. [m3 bed / hr]

        Returns:
            dd (float): [K / m]
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        Cs,dCsdt = self.HeatCapacity_Solid(t) # Cs (float): specific heat of solid particles. [kcal / kg * K] ; dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]

        q3 = self.Heat_3(z,T,t,x,y,w) # [kcal / m3 bed * hr]
        q4 = self.Heat_4(z,T,t,fs,x,y,w,p) # [kcal / m3 bed * hr]
        q5 = self.Heat_5(z,T,t,fs,x,y,w,p) # [kg / m3 bed * hr]

        dd = Az * (q3 + Cs*t*q5 + q4) / (rho_b * self.params.Fs * (Cs + t*dCsdt))
        # return np.asarray(dd).item()   
        return dd

    def dfsdz(self,z,T,t,fs,x,y,w,p):
        """differential equation of fs
        fractional reduction of iron ore

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            Fs (float): volume rate of solid particles. [m3 bed / hr]
            c_H0 (float): initial concentration of hematite. [kmol / m3 bed]

        Returns:
            dd (float): [1 / m]
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]    

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        
        weight = smooth_heaviside(t-1673,k=5)
        dd1 = Az * (R1 + R5) / 3 / self.params.Fs / self.params.c_H0
        dd2 = Az * (R5) / 3 / self.params.Fs / self.params.c_H0
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def dxdz(self,z,T,t,fs,x,y,w,p):
        """differential equation of x
        molar fraction of CO in bulk of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Returns:
            dd (float): [1 / m]
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2] 
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1673,k=5)
        dd1 = 22.4 * Az * ((1+0*x)*R1 + R7) / F
        dd2 = 22.4 * Az * (R7) / F
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def dydz(self,z,T,t,fs,x,y,w,p):
        """differential equation of y
        molar fraction of CO2 in bulk of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]
        
        Returns:
            dd (float): [1 / m]
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2] 

        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1673,k=5)
        dd1 = 22.4 * Az * ((0*y-1)*R1 - R7) / F
        dd2 = 22.4 * Az * (- R7) / F
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def dwdz(self,z,T,t,fs,x,y,w,p):
        """differential equation of w
        molar fraction of H2 in bulk of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]
        
        Returns:
            dd (float): [1 / m]
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2] 

        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1673,k=5)
        dd1 = 22.4 * Az * (0*w*R1 + R5 - R7) / F
        dd2 = 22.4 * Az * (R5 - R7) / F
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def drhobdz(self,z,T,t,fs,x,y,w,p):
        """differential equation of rho_b
        bulk density of solid particles

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:    
            Fs (float): volume rate of solid particles. [m3 bed / hr]
        
        Returns:
            dd (float): [kg / m3 bed * m]
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1673,k=5)
        dd1 = -Az * ((16+12*0)*R1+ 16*R5) / self.params.Fs
        dd2 = -Az * (16*R5) / self.params.Fs
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def dpdz(self,z,T,x,y,w,p):
        """differential equation of p
        pressure of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            epsilon (float): fractional void in bed. [-]
            F_b (float): volume rate of dry blast. [Nm3 / min]
            
        Returns:
            dd (float): [Kg / m2 * m]
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        G = F * rho / (Az * self.params.epsilon) # G (float): mass velocity of gas. [kg / m2 * hr]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        Re = self.params.d_p * G / miu
        fk = (1.75 + 150 * (1 - self.params.epsilon)) / Re

        dd = fk * (1 - self.params.epsilon) * G**2 * P_std * T / (g_c * self.params.epsilon**3 * self.params.d_p * rho * T_std * p)

        return dd

    # heat
    def Heat_2(self,z,T,t,fs,x,y,w,p):
        """
        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            q (float): [(kmol / m3 bed * hr) * (kg / m3)]
        """
        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1673,k=5)
        q1 = (1.2507*0 + 0.7261*1)*R1 + 0.7143*R5
        q2 = 0.7143*R5
        q = q1 * (1-weight) + q2 * weight

        return q
    
    def Heat_3(self,z,T,t,x,y,w):
        """
        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
        
        Operate:    
            epsilon (float): fractional void in bed. [-]
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            phi_o (float): shape factor of iron ore. [-]
            d_o (float): average diameter of particles of iron ore. [m]    
            
        Returns:
            q (float): [kcal / m3 bed * hr]
        """
        Dz = self.params.Diameter_BF(z)
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]

        C = self.HeatCapacity_Gas(T,x,y,w)[0] # C (float): specific heat of gas. [kcal / kg * K]

        G = rho * F / Az # G (float): mass velocity of gas. [kg / m2 * hr]
        Re = self.params.d_p * G / miu
        k = 0.06 # k (float): thermal conductivity of gas. [kcal / m * hr * K]
        Pr = C * miu / k
        Nu = 2.0 + 0.60 * _nonneg_sqrt(Re) * _nonneg_cbrt(Pr)
        h_p = Nu * k / self.params.d_p # h_p (float): particle-to-fluid heat transfer coefficient. [kcal / m2 * hr * K]

        q = 6 * (1-self.params.epsilon) * h_p * (T-t) / self.params.phi_o / self.params.d_p
        # print(f"hp={h_p}")
        # print(f"q3={q}")
        return q
    
    def Heat_4(self,z,T,t,fs,x,y,w,p):
        """
        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            q (float): [kcal / m3 bed * hr]
        """
        # if fs < 0.111:
        #     H1 = -7.88e3 # [kcal / kmol CO]
        #     H5 = -2.8e3 # [kcal / kmol H2]
        # elif fs < 0.333:
        #     H1 = 7.12e3
        #     H5 = 16.1e3 
        # else:
        #     H1 = -5.45e3
        #     H5 = 6.5e3

        weight1 = smooth_heaviside(fs-0.111,k=200)
        weight2 = smooth_heaviside(fs-0.333,k=200)

        H1 = np.zeros_like(z)
        H5 = np.zeros_like(z)
        mask = (fs < 0.222)
        H1[mask] = (1-weight1[mask])*-7.88e3 * 1/9 + weight1[mask]*7.12e3 * 2/9
        H5[mask] = (1-weight1[mask])*-2.8e3 * 1/9 + weight1[mask]*16.1e3 * 2/9
        H1[~mask] = (1-weight2[~mask])*7.12e3 * 2/9 + weight2[~mask]*-5.45e3 * 2/3
        H5[~mask] = (1-weight2[~mask])*16.1e3 * 2/9 + weight2[~mask]*6.5e3 * 2/3
        
        H2 = 40.8e3 # [kcal / kmol CO2]
        H3 = 31.13e3 # [kcal / kmol CO]
        H4 = 42.5e3 # [kcal / kmol CO2]
        H6 = 31.5e3 # [kcal / kmol CO]
        H7 = -9.84e3 # [kcal / kmol CO2]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]      
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1673,k=2)
        q1 = -H1*R1 -H5*R5 -H7*R7
        q2 = -H5*R5 -H7*R7
        q = (1-weight)*q1 + weight*q2

        return q    
    
    def Heat_5(self,z,T,t,fs,x,y,w,p):
        """
        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            q (float): [kg / m3 bed * hr]
        """
        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1673,k=5)
        q1 = 16*R1 + 16*R5
        q2 = 16*R5
        q = q1 * (1-weight) + q2 * weight
        # q = np.zeros_like(q)
        return q

    # reaction_rate
    def ReactionRate_1(self,z,T,t,fs,x,y,w,p):
        """overall reaction rate per unit volume of bed in reaction
        1/3 Fe2O3 + CO = 2/3 Fe + CO2

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            d_o (float): average diameter of particles of iron ore. [m]
            phi_o (float): shape factor of iron ore. [-]
            N_o (int): number of particles of iron ore per unit volume of bed. [1 / m3 bed]
            epsilon_o (float): porosity of iron ore. [-]
            
        Returns:
            r (float): reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        
        Raises:
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]

        D_CO = self.DiffusionCoefficient_CO(t,p) # D_CO (float): diffusion coefficient of CO in blast furnace gas. [m2 / hr]

        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        Re = self.params.d_o * u * rho / miu

        Sc = miu / rho / D_CO
        Sh = 2.0 + 0.55 * _nonneg_sqrt(Re) * _nonneg_cbrt(Sc)
        kf = self.TransferCoefficient_Gas(Sh,D_CO,self.params.d_o) # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

        epsilon_v = 0.53 + 0.47 * self.params.epsilon_o
        xi = 0.238 * self.params.epsilon_o + 0.04
        Ds = D_CO * epsilon_v * xi # Ds (float): intraparticle diffusion coefficient of CO in reduced iron phase. [m2 / hr]

        k = 347 * np.exp(-3460/t) # k (float): rate constant of reaction. [m / hr]

        K = self.smooth_R1(t,fs) # K (float): equilibrium constant of reaction. [-]

        xe = (x+y) / (1+K)

        _sh = _heme_shell_1_minus_fs(fs)
        r = pai * self.params.d_o**2 * self.params.phi_o**(-1) * self.params.N_o * (p/P_std) * 273 * (x-xe) / 22.4 / t / (1/kf + self.params.d_o/2*(np.power(_sh, -1.0/3.0) - 1.0)/Ds + (np.power(_sh, 2.0/3.0)*k*(1+1/K))**(-1))
        fs = np.asarray(fs)
        r = np.where(fs >= 1, 0.0, r)
        return r


    def ReactionRate_5(self,z,T,t,fs,x,y,w,p):
        """overall reaction rate per unit volume of bed in reaction
        1/3 Fe2O3 + H2 = 2/3 Fe + H2O

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            d_o (float): average diameter of particles of iron ore. [m]
            phi_o (float): shape factor of iron ore. [-]
            epsilon_o (float): porosity of iron ore. [-]
            N_o (int): number of particles of iron ore per unit volume of bed. [1 / m3 bed]

        Returns:
            r (float): reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        Raises:
        """
        t_d = _solid_T_for_diffusion_powers(t)
        D_H2 = 3.960E-6 * np.power(t_d, 1.78) / (p/P_std)    # D_H2 (float): diffusion coefficient of H2 in blast furnace gas. [m2 / hr]
        epsilon_v = 0.53 + 0.47 * self.params.epsilon_o
        xi = 0.238 * self.params.epsilon_o + 0.04
        Ds = D_H2 * epsilon_v * xi # Ds (float): intraparticle diffusion coefficient of H2 in reduced iron phase. [m2 / hr]

        Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        v = self.MolarFaction_H2O(x,y,w) # v: molar fraction of H2O in bulk of gas. [-]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        Re = self.params.d_o * u * rho / miu
        Sc = miu / rho / D_H2
        Sh = 2.0 + 0.55 * _nonneg_sqrt(Re) * _nonneg_cbrt(Sc)
        kf = self.TransferCoefficient_Gas(Sh,D_H2,self.params.d_o)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]
        k,K = self.smooth_R5(t)  # smoothed k,K

        we = (w + v) / (1+K) # we (float): molar fraction of H2 at equilibrium. [-]

        r = np.zeros_like(z)

        with np.errstate(invalid='raise'):  # 将无效值错误转为异常
            _m = ~(fs >= 1)
            _sh5 = _heme_shell_1_minus_fs(fs[_m])
            r[_m] = pai * self.params.d_o**(2) * self.params.phi_o**(-1) * self.params.N_o * 273 * (p[_m]/P_std) * (w[_m]-we[_m]) / 22.4 / t[_m] / (1/kf[_m] + self.params.d_o/2*(np.power(_sh5, -1.0/3.0) - 1.0)/Ds[_m] + (np.power(_sh5, 2.0/3.0)*k[_m]*(1+1/K[_m]))**(-1))
            r[(fs>=1)] = 0

        np.clip(r, 0, None, out=r)
        return r

    def ReactionRate_7(self,T,x,y,w,p):
        """change in moles of H2 caused by reaction
        CO + H2O = CO2 + H2

        Args:  
            T (float): temperature of gas. [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            r (float): change in moles of H2. [kmol H2 / m3 bed * hr]

        Raises:
        """
        T = np.clip(np.asarray(T, dtype=float), BVP_TEMP_MIN, BVP_TEMP_MAX)
        p = np.clip(np.asarray(p, dtype=float), BVP_P_MIN, BVP_P_MAX)
        x = np.clip(np.asarray(x, dtype=float), BVP_FRACTION_MIN, BVP_FRACTION_MAX)
        y = np.clip(np.asarray(y, dtype=float), BVP_FRACTION_MIN, BVP_FRACTION_MAX)
        w = np.clip(np.asarray(w, dtype=float), BVP_FRACTION_MIN, BVP_FRACTION_MAX)
        v = self.MolarFaction_H2O(x, y, w)  # v: molar fraction of H2O in bulk of gas. [-]

        sqrt_x = _nonneg_sqrt(x)
        sqrt_w = _nonneg_sqrt(w)
        rat = np.maximum(p / P_std / T, 0.0)
        rat_32 = rat * _nonneg_sqrt(rat)
        r_forward = (
            7.29e11
            * sqrt_x
            * v
            * rat_32
            * self.params.epsilon
            * np.exp(-67300 / R / T)
            / _nonneg_sqrt(1 + 14.158 * w * p / P_std / T)
        )
        r_backward = (
            1.386e10
            * y
            * sqrt_w
            * rat_32
            * self.params.epsilon
            * np.exp(-57000 / R / T)
            / (1 + 4.247 * x * p / P_std / T)
        )
        r = r_forward - r_backward
        return r    

    # 辅助函数
    def stable_coth(self, m):
        m = np.asarray(m)
        abs_m = np.abs(m)
        
        result = np.zeros_like(abs_m)
        mask = (abs_m > 700)
        result[mask] = np.sign(m[mask]) * 1.0
        result[~mask] = np.cosh(m[~mask]) / np.sinh(m[~mask])
        return result

        
    def smooth_R5(self, t,t0=848,k=10):
        weight = smooth_heaviside(t - t0, k=2)

        k1 = 102.78 * t * np.exp(-14900/R/t) # k (float): rate constant of reaction. [m / hr]
        K1 = np.exp(8.883 - 8475/t) # K (float): equilibrium constant of reaction. [-]

        k2 = 82.50 * t * np.exp(-15300/R/t)  
        K2 = np.exp(1.0837 - 1737.2/t)

        k = (1 - weight) * k1 + weight * k2
        K = (1 - weight) * K1 + weight * K2
        # return k,(K+eps)
        return k,K

    def smooth_R1(self, t,fs,t0=848,fs0=0.111,fs1=0.333,k=10):
        t = np.asarray(t)
        fs = np.asarray(fs)
        weight1 = smooth_heaviside(t - t0, k=20)
        weight2 = smooth_heaviside(fs - fs0, k=200)
        weight3 = smooth_heaviside(fs - fs1, k=200)

        K11 = np.exp(4.91 + 6235/t)
        K12 = np.exp(-0.7625 + 543.3/t)
        K21 = np.exp(4.91 + 6235/t)
        K22 = np.exp(2.13 - 2050/t)
        K23 = np.exp(-2.642 + 2164/t)

        K = np.zeros_like(t)
        mask1 = (fs < (fs0+fs1)/2)
        K[mask1] = (1 - weight1[mask1]) * ((1 - weight2[mask1]) * K11[mask1] + weight2[mask1] * K12[mask1]) + weight1[mask1] * ((1 - weight2[mask1]) * K21[mask1] + weight2[mask1] * K22[mask1])
        K[~mask1] = (1 - weight1[~mask1]) * K12[~mask1] + weight1[~mask1] * ((1 - weight3[~mask1]) * K22[~mask1] + weight3[~mask1] * K23[~mask1])

        # np.clip(K, None, 1e5, out=K)
        return K    # K (float): equilibrium constant of reaction. [-]
    
    # 简单变量计算函数
    def VolumeRate_Gas(self,x,y):
        """
        Args:
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]

        Returns:
            F (float): volume rate of flow of gas. [Nm3 / hr]

        Raises:
        """
        F = (self.params.SI_H2*self.params.Prod/24) / (1-x-y+eps)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        return F

    def MolarFaction_H2O(self,x,y,w):
        """
        Args:
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]

        Returns:
            v (float): molar fraction of H2O in bulk of gas. [-]

        Raises:
        """
        F = self.VolumeRate_Gas(x,y)
        v = (self.params.SI_H2*self.params.Prod/24) / F - w # v: molar fraction of H2O in bulk of gas. [-]
        return v
    
    def Density_Gas(self,x,y,w):
        """
        Args:
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]

        Returns:
            rho (float): density of blast furnace gas. [kg / Nm3]

        Raises:
        """
        rho = 0.804 - 0.714*w + 0.446*x +1.160*y # rho (float): density of blast furnace gas. [kg / Nm3]
        return rho
    
    def Viscosity_Gas(self,T):
        """
        Args:
            T (float): temperature of gas. [K]

        Returns:
            miu (float): viscosity of blast furnace gas. [kg / m * hr]

        Raises:
        """
        T0 = np.asarray(T, dtype=float)
        Tn = np.nan_to_num(
            T0,
            nan=float(BVP_TEMP_MIN),
            posinf=float(BVP_TEMP_MAX),
            neginf=float(BVP_TEMP_MIN),
        )
        Ta = np.clip(Tn, float(BVP_TEMP_MIN), float(BVP_TEMP_MAX))
        miu = 4.960e-3 * np.power(Ta, 1.5) / (Ta + 103)  # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        return miu
    
    def TransferCoefficient_Gas(self,Sh,D,d):
        """
        Args:
            Sh (float): Schmidt number of gas-film. [-]
            D (float): diffusion coefficient of gas. [m2 / hr]
            d (float): particle diameter. [m]

        Returns:
            kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

        Raises:
        """
        kf = Sh * D / d  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]
        return kf
    
    def DiffusionCoefficient_CO(self,t,p):
        """
        Args:
            t (float): temperature of solid particles(molten materials). [K]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            D_CO (float): diffusion coefficient of CO. [m2 / hr]

        Raises:
        """
        weight = smooth_heaviside(t-848,k=5)
        t_d = _solid_T_for_diffusion_powers(t)
        D_CO_1 = 2.592e-6 * np.power(t_d, 1.78) / (p/P_std)
        D_CO_2 = 2.592e-6 * np.square(t_d) / (p/P_std)
        D_CO = (1-weight) * D_CO_1 + weight * D_CO_2 # D_CO (float): diffusion coefficient of CO. [m2 / hr]
        return D_CO

    def DiffusionCoefficient_CO2(self,t,p):
        """
        Args:
            t (float): temperature of solid particles(molten materials). [K]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            D_CO2 (float): diffusion coefficient of CO2. [m2 / hr]

        Raises:
        """
        t_d = _solid_T_for_diffusion_powers(t)
        D_CO2 = 2.236E-6 * np.power(t_d, 1.78) / (p/P_std)    # D_CO2 (float): diffusion coefficient of CO2 in blast furnace gas. [m2 / hr]
        return D_CO2
    
    def HeatCapacity_Gas(self,T,x,y,w):
        """
        Args:
            T (float): temperature of gas. [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]

        Returns:
            C (float): heat capacity of gas. [kcal / kg * K]
            dCdT (float): specific heat of gas differential T. [kcal / kg * K**2]

        Raises:
        """
        v = self.MolarFaction_H2O(x,y,w)
        S1 = 6.6 + 3.9*y + 0.02*w + 0.56*v 
        S2 = (1.20 + 1.20*y - 0.39*w + 1.38*v)*1e-3
        M = 28 + 16*y - 26*w - 10*v
        C = (S1 + S2*T - 2.00e5*y/T**2) / M # C (float): specific heat of gas. [kcal / kg * K]
        dCdT = (S2 + 4e5*y/T**3) / M # dCdT (float): specific heat of gas differential T. [kcal / kg * K**2]
        return C,dCdT
    
    def HeatCapacity_Solid(self,t):
        """
        Args:
            t (float): temperature of solid particles(molten materials). [K]

        Returns:
            Cs (float): specific heat of solid particles. [kcal / kg * K]
            dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]

        Raises:
        """
        Cs = 0.1897 + 3.147e-5 * t # Cs (float): specific heat of solid particles. [kcal / kg * K]
        dCsdt = 3.147e-5 # dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]
        return Cs,dCsdt
     

class HCFurnaceModel(FurnaceModel):
    """
    高炉热量流模型
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    @staticmethod
    def clip_profile_for_hc(T, t, fs, x, y, w, rhob, p):
        """与 BVP blast_furnace_bvp 一致；在驱动层每次调用 *_hc 前对整剖面做一次约束。"""
        return clip_furnace_state_up8(T, t, fs, x, y, w, rhob, p)

    # Heat Current Method
    def Tt_hc(self,z,T,t,fs,x,y,w,p,rhob):
        """[T,t,fs,x,y,w,p,rhob]->[T_new,t_new]

        调用前请由驱动层使用 ``clip_profile_for_hc``（与 BVP 一致）。

        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, x, y, w, p, rhob (numpy.ndarray): state variables.

        Returns:
            T_new (numpy.ndarray): temperature profile of gas. [K]
            t_new (numpy.ndarray): temperature profile of coke-bed. [K]
        """
        T1in = self.params.t_in
        T2in = self.params.T_in

        Dz = self.params.Diameter_BF(z)
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        C,dCdT = self.HeatCapacity_Gas(T,x,y,w) # C (float): heat capacity of gas. [kcal / kg * K] ; dCdT (float): differential of C with T. [kcal / kg * K**2]
        Cs,dCsdt = self.HeatCapacity_Solid(t) # Cs (float): specific heat of solid particles. [kcal / kg * K] ; dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]

        G = rho * F / Az # G (float): mass velocity of gas. [kg / m2 * hr]
        Re = self.params.d_p * G / miu
        k = 0.06 # k (float): thermal conductivity of gas. [kcal / m * hr * K]
        Pr = C * miu / k
        Nu = 2.0 + 0.60 * _nonneg_sqrt(Re) * _nonneg_cbrt(Pr)
        h_p = Nu * k / self.params.d_p # h_p (float): particle-to-fluid heat transfer coefficient. [kcal / m2 * hr * K]

        KA = 6 * (1-self.params.epsilon) * h_p * Az  / self.params.phi_o / self.params.d_p  # KA (float): Heat transfer coefficient. [kcal / m * hr * K]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]        

        weight1 = smooth_heaviside(fs-0.111,k=200)
        weight2 = smooth_heaviside(fs-0.333,k=200)

        H1 = np.zeros_like(z)
        H5 = np.zeros_like(z)
        mask = (fs < 0.222)
        H1[mask] = (1-weight1[mask])*-7.88e3 * 1/9 + weight1[mask]*7.12e3 * 2/9
        H5[mask] = (1-weight1[mask])*-2.8e3 * 1/9 + weight1[mask]*16.1e3 * 2/9
        H1[~mask] = (1-weight2[~mask])*7.12e3 * 2/9 + weight2[~mask]*-5.45e3 * 2/3
        H5[~mask] = (1-weight2[~mask])*16.1e3 * 2/9 + weight2[~mask]*6.5e3 * 2/3

        H7 = -9.84e3 # [kcal / kmol CO2]

        # t<1673K
        q2 = (1.2507*0 + 0.7261*1)*R1 + 0.7143*R5
        q4 = -H1*R1 -H5*R5 -H7*R7
        q5 = 16*R1 + 16*R5

        G1 = rhob * self.params.Fs * (Cs + t*dCsdt) # solid   [kcal / hr * K]
        G2 = rho * F * (C + T*dCdT)    # gas     [kcal / hr * K]
        Q1 = Az*q4 + Az*Cs*t*q5                 # solid
        Q2 = 22.4*Az*C*q2*T + pai*Dz*self.params.U*(T-self.params.T_we) # gas       [kcal / m * hr]

        G1 = (G1[:-1] + G1[1:]) / 2
        G2 = (G2[:-1] + G2[1:]) / 2
        Q1 = (Q1[:-1] + Q1[1:]) / 2
        Q2 = (Q2[:-1] + Q2[1:]) / 2
        KA = (KA[:-1] + KA[1:]) / 2

        z_diff = np.diff(z)
        N = len(z_diff)
        A_temp,a_temp = setAa_n(N, z_diff, KA, G1, G2, T1in, T2in, Q1, Q2)
        X_temp = solve(A_temp, a_temp)

        t_new = np.asarray(X_temp[0:N+1]).reshape(-1)
        T_new = np.asarray(X_temp[(N+1):(2*N+2)]).reshape(-1)

        # plt.plot(z, T_new, label='Tnew')
        # plt.plot(z, t_new, label='tnew')
        # plt.legend()
        # plt.show()

        # plt.plot(z, T_new-T, label='Tnew-T')
        # plt.plot(z, t_new-t, label='tnew-t')
        # plt.legend()
        # plt.show()

        count = 0
        limit = HC_MAX_ITER_TT_XY_FS
        s = HC_RELAXATION
        while(
            norm(T_new - T) / norm(T) >= HC_REL_TOL_MAIN
            or norm(t_new - t) / norm(t) >= HC_REL_TOL_MAIN
        ) and (count < limit):
            count += 1
            # print("Tt_hc, count = ", count)
            T = s*T_new + (1-s)*T
            t = s*t_new + (1-s)*t

            T, t = clip_iter_temperatures(T, t)

            miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
            F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
            
            rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
            C,dCdT = self.HeatCapacity_Gas(T,x,y,w) # C (float): heat capacity of gas. [kcal / kg * K] ; dCdT (float): differential of C with T. [kcal / kg * K**2]
            Cs,dCsdt = self.HeatCapacity_Solid(t) # Cs (float): specific heat of solid particles. [kcal / kg * K] ; dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]

            G = rho * F / Az # G (float): mass velocity of gas. [kg / m2 * hr]
            Re = self.params.d_p * G / miu
            k = 0.06 # k (float): thermal conductivity of gas. [kcal / m * hr * K]
            Pr = C * miu / k
            Nu = 2.0 + 0.60 * _nonneg_sqrt(Re) * _nonneg_cbrt(Pr)
            h_p = Nu * k / self.params.d_p # h_p (float): particle-to-fluid heat transfer coefficient. [kcal / m2 * hr * K]

            KA = 6 * (1-self.params.epsilon) * h_p * Az  / self.params.phi_o / self.params.d_p  # KA (float): Heat transfer coefficient. [kcal / m * hr * K]

            R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
            R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
            R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]        

            # t<1673K
            q2 = (1.2507*0 + 0.7261*1)*R1 + 0.7143*R5
            q4 = -H1*R1 -H5*R5 -H7*R7
            q5 = 16*R1 + 16*R5

            G1 = rhob * self.params.Fs * (Cs + t*dCsdt) # solid   [kcal / hr * K]
            G2 = rho * F * (C + T*dCdT)    # gas     [kcal / hr * K]
            Q1 = Az*q4 + Az*Cs*t*q5                 # solid
            Q2 = 22.4*Az*C*q2*T + pai*Dz*self.params.U*(T-self.params.T_we) # gas       [kcal / m * hr]

            G1 = (G1[:-1] + G1[1:]) / 2
            G2 = (G2[:-1] + G2[1:]) / 2
            Q1 = (Q1[:-1] + Q1[1:]) / 2
            Q2 = (Q2[:-1] + Q2[1:]) / 2
            KA = (KA[:-1] + KA[1:]) / 2

            z_diff = np.diff(z)
            N = len(z_diff)
            A_temp,a_temp = setAa_n(N, z_diff, KA, G1, G2, T1in, T2in, Q1, Q2)
            X_temp = solve(A_temp, a_temp)

            t_new = np.asarray(X_temp[0:N+1]).reshape(-1)
            T_new = np.asarray(X_temp[(N+1):(2*N+2)]).reshape(-1)

        # print("Tt_hc, total count = ", count_out)
        return T_new, t_new

    def xy_hc(self,z,T,t,fs,x,y,w,p):
        """
        调用前请由驱动层使用 ``clip_profile_for_hc``（与 BVP 一致）。

        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, x, y, w, p (numpy.ndarray)
        Returns:
            x_new (numpy.ndarray): profile of molar fraction of CO in bulk of gas. [-]
            y_new (numpy.ndarray): profile of molar fraction of CO2 in bulk of gas. [-]
        """
        x_in = self.params.x_in
        y_in = self.params.y_in

        Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
        D_CO = self.DiffusionCoefficient_CO(t,p)
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        Re = self.params.d_o * u * rho / miu

        Sc = miu / rho / D_CO
        Sh = 2.0 + 0.55 * _nonneg_sqrt(Re) * _nonneg_cbrt(Sc)
        kf = self.TransferCoefficient_Gas(Sh,D_CO,self.params.d_o)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

        epsilon_v = 0.53 + 0.47 * self.params.epsilon_o
        xi = 0.238 * self.params.epsilon_o + 0.04
        Ds = D_CO * epsilon_v * xi # Ds (float): intraparticle diffusion coefficient of CO in reduced iron phase. [m2 / hr]

        k1 = 347 * np.exp(-3460/t) # k (float): rate constant of reaction. [m / hr]

        K1 = self.smooth_R1(t,fs) # K (float): equilibrium constant of reaction. [-]
        _sh_xy = _heme_shell_1_minus_fs(fs)
        kappa_1 = pai * self.params.d_o**2 * self.params.phi_o**(-1) * self.params.N_o * (p/P_std) * 273 / 22.4 / t / (1/kf + self.params.d_o/2*(np.power(_sh_xy, -1.0/3.0) - 1.0)/Ds + (np.power(_sh_xy, 2.0/3.0)*k1*(1+1/K1))**(-1))
        
        kappa_1[fs>=1] = 0
        
        KA = Az*kappa_1 # transfer coefficient [Nm3 / m * hr]

        R7 = self.ReactionRate_7(T,x,y,w,p)

        G1 = F/22.4 * (1+K1)/K1  # G1 (float): capacity flow of x. [kmol / hr]
        G2 = F/22.4 * (1+K1)/K1  # G2 (float): capacity flow of y. [kmol / hr]

        Q1 = Az*((K1-1)/(1+K1)*kappa_1*y + R7) * (1+K1)/K1
        Q2 = Az*(-(K1-1)/(1+K1)*kappa_1*y - R7) * (1+K1)/K1

        G1 = (G1[:-1] + G1[1:]) / 2
        G2 = (G2[:-1] + G2[1:]) / 2
        Q1 = (Q1[:-1] + Q1[1:]) / 2
        Q2 = (Q2[:-1] + Q2[1:]) / 2
        KA = (KA[:-1] + KA[1:]) / 2

        z_diff = np.diff(z)
        N = len(z_diff)
        A_temp, a_temp = setAa_s(N, z_diff, KA, G1, G2, x_in, y_in, Q1, Q2)
        X_temp = solve(A_temp, a_temp)
        x_new = np.asarray(X_temp[0:N+1]).reshape(-1)
        y_new = np.asarray(X_temp[(N+1):(2*N+2)]).reshape(-1)


        count = 0
        limit = HC_MAX_ITER_TT_XY_FS
        s = HC_RELAXATION
        while(
            norm(x_new - x) / norm(x) >= HC_REL_TOL_MAIN
            or norm(y_new - y) / norm(y) >= HC_REL_TOL_MAIN
        ) and (count < limit):
            count += 1
            # print("xy_hc, count_out = ", count_out)
            # print("norm(x_new-x)/norm(x) = ", norm(x_new-x)/norm(x))
            # print("norm(y_new-y)/norm(y) = ", norm(y_new-y)/norm(y))
            x = s*x_new + (1-s)*x
            y = s*y_new + (1-s)*y

            x = np.clip(x, BVP_FRACTION_MIN, BVP_FRACTION_MAX)
            y = np.clip(y, BVP_FRACTION_MIN, np.minimum(BVP_FRACTION_MAX, 1.0 - x - eps))

            F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
            u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
            rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
            Re = self.params.d_o * u * rho / miu

            Sc = miu / rho / D_CO
            Sh = 2.0 + 0.55 * _nonneg_sqrt(Re) * _nonneg_cbrt(Sc)
            kf = self.TransferCoefficient_Gas(Sh,D_CO,self.params.d_o)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

            _sh_xy = _heme_shell_1_minus_fs(fs)
            kappa_1 = pai * self.params.d_o**2 * self.params.phi_o**(-1) * self.params.N_o * (p/P_std) * 273 / 22.4 / t / (1/kf + self.params.d_o/2*(np.power(_sh_xy, -1.0/3.0) - 1.0)/Ds + (np.power(_sh_xy, 2.0/3.0)*k1*(1+1/K1))**(-1))
            kappa_1[fs>=1] = 0

            KA = Az*kappa_1 # transfer coefficient [Nm3 / m * hr]

            R7 = self.ReactionRate_7(T,x,y,w,p)

            G1 = F/22.4 * (1+K1)/K1  # G1 (float): capacity flow of x. [kmol / hr]
            G2 = F/22.4 * (1+K1)/K1  # G2 (float): capacity flow of y. [kmol / hr]

            Q1 = Az*((K1-1)/(1+K1)*kappa_1*y + R7) * (1+K1)/K1
            Q2 = Az*(-(K1-1)/(1+K1)*kappa_1*y - R7) * (1+K1)/K1

            G1 = (G1[:-1] + G1[1:]) / 2
            G2 = (G2[:-1] + G2[1:]) / 2
            Q1 = (Q1[:-1] + Q1[1:]) / 2
            Q2 = (Q2[:-1] + Q2[1:]) / 2
            KA = (KA[:-1] + KA[1:]) / 2

            z_diff = np.diff(z)
            N = len(z_diff)
            A_temp, a_temp = setAa_s(N, z_diff, KA, G1, G2, x_in, y_in, Q1, Q2)
            X_temp = solve(A_temp, a_temp)
            x_new = np.asarray(X_temp[0:N+1]).reshape(-1)
            y_new = np.asarray(X_temp[(N+1):(2*N+2)]).reshape(-1)

            # plt.plot(z, x_new, label='xnew')
            # plt.plot(z, y_new, label='ynew')
            # plt.legend()
            # plt.show()

            # plt.plot(z, x_new-x, label='xnew-x')
            # plt.plot(z, y_new-y, label='ynew-y')
            # plt.legend()
            # plt.show()
        
        return x_new, y_new

    def w_hc(self,z,T,t,fs,x,y,w,p):
        """
        
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, fl, x, y, w, p (numpy.ndarray)
        Returns:
            w_new (numpy.ndarray): profile of molar fraction of H2 in bulk of gas. [-]
        """
        w_in = self.params.w_in

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2] 
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        Re = self.params.d_o * u * rho / miu
        t_d = _solid_T_for_diffusion_powers(t)
        D_H2 = 3.960E-6 * np.power(t_d, 1.78) / (p/P_std) # D_H2 (float): diffusion coefficient of H2 in blast furnace gas. [m2 / hr]
        Sc = miu / rho / D_H2
        Sh = 2.0 + 0.55 * _nonneg_sqrt(Re) * _nonneg_cbrt(Sc)
        kf = self.TransferCoefficient_Gas(Sh,D_H2,self.params.d_o)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]
        epsilon_v = 0.53 + 0.47 * self.params.epsilon_o
        xi = 0.238 * self.params.epsilon_o + 0.04
        Ds = D_H2 * epsilon_v * xi # Ds (float): intraparticle diffusion coefficient of H2 in reduced iron phase. [m2 / hr]
        k,K = self.smooth_R5(t)  # smoothed k,K

        _sh_w = _heme_shell_1_minus_fs(fs)
        kappa_5 = pai * self.params.d_o**(2) * self.params.phi_o**(-1) * self.params.N_o * (p/P_std) * 273 / 22.4 / t / (1/kf + self.params.d_o/2*(np.power(_sh_w, -1.0/3.0) - 1.0)/Ds + (np.power(_sh_w, 2.0/3.0)*k*(1+1/K))**(-1))

        R7 = self.ReactionRate_7(T,x,y,w,p)

        a_list = 22.4*Az*kappa_5/F
        b_list = 22.4*Az*(- kappa_5*(self.params.SI_H2*self.params.Prod/24)/F/(1+K) - R7) / F

        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p)
        a_list[R5<=0] = 0
        b_list[R5<=0] = 22.4*Az[R5<=0]*( - R7[R5<=0]) / F[R5<=0]

        a_list = (a_list[1:] + a_list[:-1]) / 2
        b_list = (b_list[1:] + b_list[:-1]) / 2

        z_diff = np.diff(z)
        N = len(z_diff)
        A_temp,a_temp = setAa_linear_n(N, z_diff, w_in, a_list, b_list)
        X_temp = solve(A_temp, a_temp)
        w_new = np.asarray(X_temp).reshape(-1)

        # plt.plot(z, w_new, label='wnew')
        # plt.plot(z, w, label='w')
        # plt.legend()
        # plt.show()

        count = 0
        limit = HC_MAX_ITER_W
        s = HC_RELAXATION
        while (norm(w_new - w) / norm(w) >= HC_REL_TOL_TIGHT) and (count < limit):
            count += 1
            # print("w_hc, count = ", count)
            # print("norm(b-Ax)/norm(b) = ", norm(a_temp - A_temp@X_previous) / norm(a_temp))
            w = s*w_new + (1-s)*w

            rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
            Re = self.params.d_o * u * rho / miu
            Sc = miu / rho / D_H2
            Sh = 2.0 + 0.55 * _nonneg_sqrt(Re) * _nonneg_cbrt(Sc)
            kf = Sh * D_H2 / self.params.d_o  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

            _sh_w = _heme_shell_1_minus_fs(fs)
            kappa_5 = pai * self.params.d_o**(2) * self.params.phi_o**(-1) * self.params.N_o * (p/P_std) * 273 / 22.4 / t / (1/kf + self.params.d_o/2*(np.power(_sh_w, -1.0/3.0) - 1.0)/Ds + (np.power(_sh_w, 2.0/3.0)*k*(1+1/K))**(-1))

            R7 = self.ReactionRate_7(T,x,y,w,p)

            a_list = 22.4*Az*kappa_5/F
            b_list = 22.4*Az*(- kappa_5*(self.params.SI_H2*self.params.Prod/24)/F/(1+K) - R7) / F

            R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p)
            a_list[R5<=0] = 0
            b_list[R5<=0] = 22.4*Az[R5<=0]*( - R7[R5<=0]) / F[R5<=0]
            
            a_list = (a_list[1:] + a_list[:-1]) / 2
            b_list = (b_list[1:] + b_list[:-1]) / 2

            A_temp,a_temp = setAa_linear_n(N, z_diff, w_in, a_list, b_list)
            X_temp = solve(A_temp, a_temp)
            w_new = np.asarray(X_temp).reshape(-1)



        # print("norm(b-Ax)/norm(b) = ", norm(a_temp - A_temp@X_previous) / norm(a_temp))
        # print("w_hc, total count = ", count)
        return w_new

    def p_hc(self,z,T,x,y,w,p):
        """
        
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, x, y, w, p (numpy.ndarray)
        Returns:
            p_new (numpy.ndarray): profile of pressure of gas. [Kg / m2]
        """
        p2_in = self.params.p_in**2

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        G = F * rho / (Az * self.params.epsilon) # G (float): mass velocity of gas. [kg / m2 * hr]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        Re = self.params.d_p * G / miu
        fk = (1.75 + 150 * (1 - self.params.epsilon)) / Re

        a_list = fk * (1 - self.params.epsilon) * G**2 * P_std * T / (g_c * self.params.epsilon**3 * self.params.d_p * rho * T_std)
        a_list = (a_list[1:] + a_list[:-1]) / 2

        z_diff = np.diff(z)
        N = len(z_diff)
        A_temp,a_temp = setAa_p(N, z_diff, p2_in, a_list)
        X_temp = solve(A_temp, a_temp)
        # print(X_temp.shape)
        p2_new = np.asarray(X_temp).reshape(-1)
        p_new = _nonneg_sqrt(p2_new)

        return p_new

    def fs_hc(self,z,T,t,fs,x,y,w,p):
        """
        
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, x, y, w, p (numpy.ndarray)
        Returns:
            fs_new (numpy.ndarray): profile of fraction of reduction of iron ore. [-]
        """
        fs_in = self.params.fs_in

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p)
        # R3 = ReactionRate_3(t,fs)
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p)

        dd = Az * (R1 + R5) / 3 / self.params.Fs / self.params.c_H0

        a_list = dd
        a_list = (a_list[1:] + a_list[:-1]) / 2

        z_diff = np.diff(z)
        N = len(z_diff)
        A_temp,a_temp = setAa_constant_s(N, z_diff, fs_in, a_list)
        X_temp = solve(A_temp, a_temp)
        fs_new = np.asarray(X_temp).reshape(-1)

        # plt.plot(z, fs_new, label='fsnew')
        # plt.plot(z, fs, label='fs')
        # plt.legend()
        # plt.show()

        count = 0
        limit = HC_MAX_ITER_TT_XY_FS
        s = HC_RELAXATION
        while (norm(fs_new - fs) / norm(fs) >= HC_REL_TOL_TIGHT) and (count < limit):
            count += 1
            # print("norm(b-Ax)/norm(b) = ", norm(a_temp - A_temp@X_previous) / norm(a_temp))
            fs = s*fs_new + (1-s)*fs
            fs = np.clip(fs, BVP_FRACTION_MIN, BVP_FRACTION_MAX)

            R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p)
            R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p)

            dd = Az * (R1 + R5) / 3 / self.params.Fs / self.params.c_H0

            a_list = dd
            a_list = (a_list[1:] + a_list[:-1]) / 2

            z_diff = np.diff(z)
            N = len(z_diff)
            A_temp,a_temp = setAa_constant_s(N, z_diff, fs_in, a_list)
            X_temp = solve(A_temp, a_temp)
            # print(X_temp.shape)
   
            fs_new = np.asarray(X_temp).reshape(-1)
        # print("norm(b-Ax)/norm(b) = ", norm(a_temp - A_temp@X_previous) / norm(a_temp))
        # print("fs_hc total count = ", count)
        return fs_new

    def rhob_hc(self,z,T,t,fs,x,y,w,p,rhob):
        """
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, x, y, w, p, rhob (numpy.ndarray)
        Returns:
            rhob_new (numpy.ndarray): profile of . [kg / m3]
        """
        rhob_in = self.params.rhob_in

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        dd = -Az * ((16+12*0)*R1 + 16*R5) / self.params.Fs

        a_list = dd
        a_list = (a_list[1:] + a_list[:-1]) / 2

        z_diff = np.diff(z)
        N = len(z_diff)
        A_temp,a_temp = setAa_constant_s(N, z_diff, rhob_in, a_list)
        X_temp = solve(A_temp, a_temp)
        # print(X_temp.shape)
        rhob_new = np.asarray(X_temp).reshape(-1)

        return rhob_new
    
    def test_hc_4n4(self):
        """
        双循环
        """
        logging.info("测试 上半部分模型 hc_4n4")
        # params = load_parameters("default_case")   # 调用已保存的参数
        # params2 = quick_modify(params, 
        #                     case_name="my_design",
        #                     initial_mesh=2000)
        model = HCFurnaceModel(self.params)

        # 1. 初值设置（分段线性，由参数类生成）
        y_init, H_ctrl = model.params.initial_bvp_guess()
        T, t, fs, x, y, w, rhob, p = y_init
        H0, HH = H_ctrl[0], H_ctrl[-1]
        z_guess = np.linspace(H0, HH, model.params.initial_mesh)

        T, t, fs, x, y, w, rhob, p = HCFurnaceModel.clip_profile_for_hc(
            T, t, fs, x, y, w, rhob, p
        )
        T_new, t_new = model.Tt_hc(z_guess, T, t, fs, x, y, w, p, rhob)
        x_new, y_new = model.xy_hc(z_guess, T, t, fs, x, y, w, p)

        RE_T = norm(T_new - T)/norm(T)
        RE_t = norm(t_new - t)/norm(t)
        RE_x = norm(x_new - x)/norm(x)
        RE_y = norm(y_new - y)/norm(y)
        count = 0
        while (
            RE_T >= HC_REL_TOL_MAIN
            or RE_t >= HC_REL_TOL_MAIN
            or RE_x >= HC_REL_TOL_MAIN
            or RE_y >= HC_REL_TOL_MAIN
        ) and (count < HC_MAX_ITER_TEST_FIRST_UP):
            count += 1
            # print("first loop count = ", count)
            # print("relative error of T = ", RE_T)
            # print("relative error of t = ", RE_t)
            # print("relative error of x = ", RE_x)
            # print("relative error of y = ", RE_y)

            T = T_new
            t = t_new
            x = x_new
            y = y_new

            T, t, fs, x, y, w, rhob, p = HCFurnaceModel.clip_profile_for_hc(
                T, t, fs, x, y, w, rhob, p
            )
            T_new, t_new = model.Tt_hc(z_guess, T, t, fs, x, y, w, p, rhob)
            x_new, y_new = model.xy_hc(z_guess, T, t, fs, x, y, w, p)

            RE_T = norm(T_new - T)/norm(T)
            RE_t = norm(t_new - t)/norm(t)
            RE_x = norm(x_new - x)/norm(x)
            RE_y = norm(y_new - y)/norm(y)

        # 外层循环：wfsprhob
        T, t, fs, x, y, w, rhob, p = HCFurnaceModel.clip_profile_for_hc(
            T, t, fs, x, y, w, rhob, p
        )
        w_new = model.w_hc(z_guess, T, t, fs, x, y, w, p)
        fs_new = model.fs_hc(z_guess, T, t, fs, x, y, w, p)
        p_new = model.p_hc(z_guess, T, x, y, w, p)
        rhob_new = model.rhob_hc(z_guess, T, t, fs, x, y, w, p, rhob)
        RE_w = norm(w_new - w)/norm(w)
        RE_fs = norm(fs_new - fs)/norm(fs)
        RE_p = norm(p_new - p)/norm(p)
        RE_rhob = norm(rhob_new - rhob)/norm(rhob)

        count_out = 0
        while (
            RE_w >= HC_REL_TOL_MAIN
            or RE_fs >= HC_REL_TOL_MAIN
            or RE_p >= HC_REL_TOL_MAIN
            or RE_rhob >= HC_REL_TOL_MAIN
        ) and (count_out < HC_MAX_ITER_TEST_OUTER_UP):
            count_out += 1
            print("count_out = ", count_out)
            # 内层循环：wfsprhob
            count_in = 0
            while (
                RE_w >= HC_REL_TOL_MAIN
                or RE_fs >= HC_REL_TOL_MAIN
                or RE_p >= HC_REL_TOL_MAIN
                or RE_rhob >= HC_REL_TOL_MAIN
            ) and (count_in < HC_MAX_ITER_TEST_NESTED_INNER):
                
                count_in += 1
                # print("count_in = ", count_in)
                # print("relative error of w = ", RE_w)
                # print("relative error of fs = ", RE_fs)
                # print("relative error of p = ", RE_p)
                # print("relative error of rhob = ", RE_rhob)
                w = w_new
                fs = fs_new
                p = p_new
                rhob = rhob_new
                T, t, fs, x, y, w, rhob, p = HCFurnaceModel.clip_profile_for_hc(
                    T, t, fs, x, y, w, rhob, p
                )
                w_new = model.w_hc(z_guess, T, t, fs, x, y, w, p)
                fs_new = model.fs_hc(z_guess, T, t, fs, x, y, w, p)
                p_new = model.p_hc(z_guess, T, x, y, w, p)
                rhob_new = model.rhob_hc(z_guess, T, t, fs, x, y, w, p, rhob)
                RE_w = norm(w_new - w)/norm(w)
                RE_fs = norm(fs_new - fs)/norm(fs)
                RE_p = norm(p_new - p)/norm(p)
                RE_rhob = norm(rhob_new - rhob)/norm(rhob)
            # print("count_in = ", count_in)
            # print("relative error of w = ", RE_w)
            # print("relative error of fs = ", RE_fs)
            # print("relative error of p = ", RE_p)
            # print("relative error of rhob = ", RE_rhob)

            # 内层循环：Ttxyfs
            T, t, fs, x, y, w, rhob, p = HCFurnaceModel.clip_profile_for_hc(
                T, t, fs, x, y, w, rhob, p
            )
            T_new, t_new = model.Tt_hc(z_guess, T, t, fs, x, y, w, p, rhob)
            x_new, y_new = model.xy_hc(z_guess, T, t, fs, x, y, w, p)

            # plt.plot(z_guess, T_new, label='T_new')
            # plt.plot(z_guess, t_new, label='t_new')
            # plt.legend()
            # plt.show()
            # plt.plot(z_guess, x_new, label='x_new')
            # plt.plot(z_guess, y_new, label='y_new')
            # plt.legend()
            # plt.show()

            RE_T = norm(T_new - T)/norm(T)
            RE_t = norm(t_new - t)/norm(t)
            RE_x = norm(x_new - x)/norm(x)
            RE_y = norm(y_new - y)/norm(y)

            count_in = 0
            while (
                RE_T >= HC_REL_TOL_MAIN
                or RE_t >= HC_REL_TOL_MAIN
                or RE_x >= HC_REL_TOL_MAIN
                or RE_y >= HC_REL_TOL_MAIN
            ) and (count_in < HC_MAX_ITER_TEST_NESTED_INNER):
                count_in += 1
                # print("count_in = ", count_in)
                # print("relative error of T = ", RE_T)
                # print("relative error of t = ", RE_t)
                # print("relative error of x = ", RE_x)
                # print("relative error of y = ", RE_y)

                T = T_new
                t = t_new
                x = x_new
                y = y_new

                T, t, fs, x, y, w, rhob, p = HCFurnaceModel.clip_profile_for_hc(
                    T, t, fs, x, y, w, rhob, p
                )
                T_new, t_new = model.Tt_hc(z_guess, T, t, fs, x, y, w, p, rhob)
                x_new, y_new = model.xy_hc(z_guess, T, t, fs, x, y, w, p)
                RE_T = norm(T_new - T)/norm(T)
                RE_t = norm(t_new - t)/norm(t)
                RE_x = norm(x_new - x)/norm(x)
                RE_y = norm(y_new - y)/norm(y)

            # print("count_in = ", count_in)
            # print("relative error of T = ", RE_T)
            # print("relative error of t = ", RE_t)
            # print("relative error of x = ", RE_x)
            # print("relative error of y = ", RE_y)

            T, t, fs, x, y, w, rhob, p = HCFurnaceModel.clip_profile_for_hc(
                T, t, fs, x, y, w, rhob, p
            )
            w_new = model.w_hc(z_guess, T, t, fs, x, y, w, p)
            fs_new = model.fs_hc(z_guess, T, t, fs, x, y, w, p)
            p_new = model.p_hc(z_guess, T, x, y, w, p)
            rhob_new = model.rhob_hc(z_guess, T, t, fs, x, y, w, p, rhob)
            RE_w = norm(w_new - w)/norm(w)
            RE_fs = norm(fs_new - fs)/norm(fs)
            RE_p = norm(p_new - p)/norm(p)
            RE_rhob = norm(rhob_new - rhob)/norm(rhob)

        logging.info("final relative error:")
        logging.info(f"relative error of T = {RE_T}")
        logging.info(f"relative error of t = {RE_t}")
        logging.info(f"relative error of x = {RE_x}")
        logging.info(f"relative error of y = {RE_y}")
        logging.info(f"relative error of w = {RE_w}")
        logging.info(f"relative error of p = {RE_p}")
        logging.info(f"relative error of fs = {RE_fs}")
        logging.info(f"relative error of rhob = {RE_rhob}")
        re_list = (RE_T, RE_t, RE_x, RE_y, RE_w, RE_fs, RE_p, RE_rhob)
        hc_max_re_final = float(max(re_list))
        hc_converged = all(re < HC_REL_TOL_MAIN for re in re_list)
        if not hc_converged:
            logging.warning(
                "test_hc_4n4 未达 HC_REL_TOL_MAIN=%s：max(RE)=%s",
                HC_REL_TOL_MAIN,
                hc_max_re_final,
            )
        # 结果绘图
        y_plot = [T_new, t_new, fs_new, x_new, y_new, w_new, rhob_new, p_new]
        plt.figure(figsize=(12, 8))
        variables = ['T', 't', 'fs', 'x', 'y', 'w', 'rhob', 'p']
        for i in range(8):
            plt.subplot(3, 3, i+1)
            plt.plot(z_guess, y_plot[i])
            plt.ylabel(variables[i])
            plt.xlabel('z')
        plt.tight_layout()
        # plt.show()

        # 剖面 CSV 由网格无关性 / 初值范围测试脚本写出
        df = pd.DataFrame(np.vstack((z_guess, y_plot)).T, columns=['z'] + variables)
        self.last_hc_profile_df = df
        # df.to_csv('test_hc_4n4_1e-3_UP_loop_debug.csv', index=False)

        self.results = {
            "case_name": self.params.case_name,
            "T_out": T_new[0],
            "t_out": t_new[-1],
            "fs_out": fs_new[-1],
            "x_out": x_new[0],
            "y_out": y_new[0],
            "w_out": w_new[0],    
            "rhob_out": rhob_new[-1],    
            "p_bottom": p_new[-1],
            "hc_converged": hc_converged,
            "hc_max_re_final": hc_max_re_final,
        }

        return self.results