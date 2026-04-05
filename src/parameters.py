# parameters.py
import numpy as np

from constant import pai, M_Fe, M_O

class FurnaceParameters:
    """高炉参数类"""
    
    def __init__(self, case_name="default_case"):
        self.case_name = case_name
        
        # 几何参数 Design parameters
        self.D0 = 1682e-3 # [m] diameter of stockline
        self.D1 = 1717e-3 # [m] diameter of hearth
        self.Db = 2403e-3 # [m] diameter of bosh
        self.Ls = 3744e-3 # [m] height of shaft
        self.La = 844e-3 # [m] height of bosh
        self.Lb = 2168e-3 # [m] height between bosh and tuyere
        
        # 操作参数 Operation parameters
        self.epsilon = 0.3 # [-] fractional void in bed
        self.p0 = 2040 # [Kg / m2] top pressure
        self.d_p = 0.009 # [m] diameter of solid particles, for h_p calculation
        self.T_we = 35 + 273 # exit tempereture of cooling water. [K]
        self.U = 10 # [kcal / m2 * hr * K] estimated value of overall heat transfer coefficient based on inner surface area of furnace-wall.
        self.W_o = 4392 # mass rate of flow of iron ore. [kg(ore) / hr]
        self.W_c = 0 # mass rate of flow of coke. [kg(coke) / hr]
        self.W_L = 0 # mass rate of flow of limestone. [kg(limestone) / hr]
        self.alpha_Fe = 0.673 # [-] weight fraction of Fe in iron ore

        self.HM_Fe = 0.99 # molar fraction of Fe in pig iron. [-]
        self.HI_O2 = 188 # hearth injection rate of O2 [Nm3 / tHM]
        self.SI = 1141 # shaft injection rate [Nm3 / tHM]
        self.SI_H2 = 0.7*self.SI # shaft injection rate of H2 [Nm3 / tHM]
        self.SI_CO = self.SI - self.SI_H2 # shaft injection rate of CO [Nm3 / tHM]

        # 材料参数 Material parameters
        self.d_o = 0.009 # [m] diameter of ore
        self.phi_o = 0.8 # [-] shape factor of ore
        self.epsilon_o = 0.20 # [-] porosity of ore
        self.rho_po = 2200 # [kg/m3] apparent density of solid particles of ore
        # 焦炭/石灰石仅用表观密度参与 F_c、F_L；颗粒级 N_c、N_L 未接入控制方程
        rho_pc = 477 # [kg/m3] apparent density of coke particles
        rho_pL = 1599 # [kg/m3] apparent density of limestone particles
        
        # Calculated parameters
        self.c_H0 = self.alpha_Fe*self.rho_po/M_Fe / 2 # initial concentration of hematite. [kmol / m3 bed]
        self.F_o = self.W_o / self.rho_po # [m3 bed / hr] volume rate of ore
        self.F_c = self.W_c / rho_pc # [m3 bed / hr] volume rate of coke
        self.F_L = self.W_L / rho_pL # [m3 bed / hr] volume rate of limestone
        self.Fs = self.F_o + self.F_c + self.F_L # volume rate of solid particles. [m3 bed / hr]
        self.N_o = (1-self.epsilon) / (4/3*pai*(self.d_o/2)**3) * self.F_o/self.Fs # [1/m3 bed] number of particles per unit volume of bed
        self.Prod = self.W_o * (2*M_Fe/(2*M_Fe+3*M_O)) / self.HM_Fe / 1000 * 24 # [tHM / d] productivity of furnace.
        self.H2_input = self.SI_H2*self.Prod/24 # [Nm3 / hr] shaft injection rate of H2
        self.CO_input = self.SI_CO*self.Prod/24 # [Nm3 / hr] shaft injection rate of CO

        # 边界条件 Boundary conditions
        self.T_in = 1223 # [K] inlet temperature of gas
        self.t_in = 298 # [K] inlet temperature of solid
        self.fs_in = 0.0 # [-] inlet reduction faction of ore
        self.x_in = 0.437 # [-] inlet mole fraction of CO
        self.y_in = 0.057 # [-] inlet mole fraction of CO2
        self.w_in = 0.506 # [-] inlet mole fraction of H2
        self.rhob_in = 2200 # [kg / m3 bed] inlet density of bed
        self.p_in = self.p0 # [Kg / m2] top pressure


        # 初始节点 Initial nodes
        self.H0 = 0.0 # [m] height of the starting point of calculation
        # self.H1 = 1
        # self.H2 = 2
        # self.H3 = 3
        self.HH = 4.166 # [m] height of the end point of calculation

        # --- 原多控制点参考初值（由 CSV 剖面）已停用，改用仅首尾两点的整段线性初值 ---
        # # 节点初值 Node initial values = [T, t, fs, x, y, w, rhob, p]（与上半部 BVP 状态次序一致）
        # # 在固定控制高度 z=0,1,2,3,4.166 m 上，由参考剖面 data/initial_case_bvp_0.0-4.2m_loop.csv 线性插值得到
        # # 0 m
        # self.value0 = [526.9271092796849, 298.0, 0.0, 0.22880648698891, 0.2446938426052689, 0.3742346598870991, 2200.0, 2040.0]
        # # 1 m
        # self.value1 = [768.5450520721031, 768.4266955055139, 0.07247458649151524, 0.25609021016926936, 0.21741011942490954, 0.3742346598871008, 2154.011768817427, 3949.2486168350724]
        # # 2 m
        # self.value2 = [832.4113278055885, 831.8621062104935, 0.20253659472790633, 0.305053246742327, 0.16844708285185195, 0.37423465988713195, 2071.4818505053695, 5098.50006379039]
        # # 3 m
        # self.value3 = [924.6551248436033, 925.0019598284767, 0.38661171632160135, 0.36672667507766266, 0.10677365451651624, 0.3818580003238633, 1954.678296920387, 5995.62589597534]
        # # 4.166 m
        # self.valueH = [1223.0155506569308, 1222.5322195493166, 1.0, 0.4735003295941787, 0.0, 0.506, 1565.4571331339614, 7052.175003843975]

        # 首尾控制点 [T, t, fs, x, y, w, rhob, p]：与 bc 在两端一致的量直接取边界；其余为简单物理占位，整段线性
        # z=H0：ya 约束 t,fs,rhob,p；z=HH：yb 约束 T,x,y,w
        self.bvp_guess_at_H0 = [
            self.t_in,
            self.t_in,
            self.fs_in,
            self.x_in,
            self.y_in,
            self.w_in,
            self.rhob_in,
            self.p_in,
        ]
        self.bvp_guess_at_HH = [
            self.T_in,
            self.T_in,
            1.0,
            self.x_in,
            self.y_in,
            self.w_in,
            min(self.rhob_in, 1600.0),
            self.p_in + (7052.0 - 2040.0),
        ]


        # 数值参数 Numerical parameters
        self.initial_mesh = 100


    def Diameter_BF(self, z):
        """Diameter of blast furnace

        Args:
            z (float): height from the stock line. [m]
        
        Returns:
            D (float): Diameter of blast furnace. [m]
        """
        # z = z * 25.25/23 # 20251105

        z = np.asarray(z)
        D = np.zeros_like(z)
        mask1 = (z <= self.Ls)
        mask2 = (z > self.Ls) & (z <= self.Ls+self.La)
        mask3 = (z > self.Ls+self.La) & (z <= self.Ls+self.La+self.Lb)
        mask4 = (z > self.Ls+self.La+self.Lb)
        # D[mask1] = D0 + 2*z[mask1]/np.tan(omega_1)
        D[mask1] = self.D0 + z[mask1]*((self.Db-self.D0)/self.Ls)
        D[mask2] = self.Db
        D[mask3] = self.Db - (z[mask3]-self.Ls-self.La)/self.Lb*(self.Db-self.D1)
        D[mask4] = self.D1

        return D # [m2]

    @staticmethod
    def _linear_interp_on_mesh(x_control, y_control, num_points):
        """在 [H0, HH] 上均匀取 num_points 个点，对 y 做分段线性插值。"""
        x_control = np.asarray(x_control, dtype=float)
        y_control = np.asarray(y_control, dtype=float)
        x_out = np.linspace(x_control[0], x_control[-1], int(num_points))
        return np.interp(x_out, x_control, y_control)

    def control_heights_and_node_values(self):
        """控制高度与各点初值；每点为 [T, t, fs, x, y, w, rhob, p]。"""
        # return (
        #     [self.H0, self.H1, self.H2, self.H3, self.HH],
        #     [self.value0, self.value1, self.value2, self.value3, self.valueH],
        # )
        return (
            [self.H0, self.HH],
            [self.bvp_guess_at_H0, self.bvp_guess_at_HH],
        )

    def initial_bvp_guess(self, num_points=None):
        """
        由首尾两控制点沿高度线性插值，得到 BVP 初值。

        Returns:
            y_guess: ndarray, shape (8, num_points)，行次序 T,t,fs,x,y,w,rhob,p
            H_ctrl: list of control heights（与 solve_bvp 区间节点一致）
        """
        if num_points is None:
            num_points = self.initial_mesh
        H_ctrl, vals = self.control_heights_and_node_values()
        mat = np.asarray(vals, dtype=float)
        rows = [
            self._linear_interp_on_mesh(H_ctrl, mat[:, j], num_points)
            for j in range(mat.shape[1])
        ]
        return np.vstack(rows), H_ctrl


def create_standard_case(case_type="default"):
    """创建标准算例"""
    if case_type == "O2_rich_0.03":
        params = FurnaceParameters("O2_rich_0.03")

        params.W_o = 264e3 # mass rate of flow of iron ore. [kg(ore) / hr]
        params.W_c = 77.8e3 # mass rate of flow of coke. [kg(coke) / hr]
        params.W_L = 12.858e3 # mass rate of flow of limestone. [kg(limestone) / hr]
    
    elif case_type == "O2_rich_0.07":
        params = FurnaceParameters("O2_rich_0.07")

        params.W_o = 309e3 # mass rate of flow of iron ore. [kg(ore) / hr]
        params.W_c = 90.2e3 # mass rate of flow of coke. [kg(coke) / hr]
        params.W_L = 15.049e3 # mass rate of flow of limestone. [kg(limestone) / hr]
    
    else:  # default
        params = FurnaceParameters(case_type)
    
    return params

def quick_modify(base_params, **changes):
    """快速修改参数（复制一份并覆盖字段；与 parameters_DOWN.FurnaceParameters 兼容）。"""
    new_params = type(base_params)()
    for key, value in base_params.__dict__.items():
        setattr(new_params, key, value)
    for key, value in changes.items():
        if hasattr(new_params, key):
            setattr(new_params, key, value)
    return new_params