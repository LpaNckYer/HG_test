# parameters_DOWN.py
import numpy as np

from constant import pai, M_Fe, M_O
from parameters import quick_modify

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
        self.p0 = 10200 # [Kg / m2] top pressure
        self.d_p = 0.009 # [m] diameter of solid particles, for h_p calculation
        self.T_we = 1000 + 273 # exit tempereture of cooling water. [K]
        self.U = 4 # [kcal / m2 * hr * K] estimated value of overall heat transfer coefficient based on inner surface area of furnace-wall.
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
        self.T_in = 1273 # [K] inlet temperature of gas
        self.t_in = 1200 # [K] inlet temperature of solid
        self.fs_in = 1 # [-] inlet reduction faction of ore
        self.x_in = 1 - 1e-15 # [-] inlet mole fraction of CO
        self.y_in = 1e-15 # [-] inlet mole fraction of CO2
        self.w_in = 0 # [-] inlet mole fraction of H2
        self.rhob_in = 1700 # [kg / m3 bed] inlet density of bed
        self.p_in = self.p0 # [Kg / m2] top pressure

        # 初始节点 Initial nodes
        self.H0 = 4.166 # [m] height of the starting point of calculation
        self.H1 = 5.872 # [m]
        self.HH = 5.872 # [m] height of the end point of calculation

        # 节点初值 Node initial values = [T, t, fs, x, rhob, p]（与下半部 BVP 状态次序一致）
        # 4.166 m
        self.value0 = [1223, 1200, 1, 0.8, 1700, 10200]
        # 5 m
        self.value1 = [1253, 1250, 1, 0.9, 1670, 12000]
        # 5.872 m
        self.valueH = [1273, 1270, 1, 1.0, 1640, 14000]


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
        x_control = np.asarray(x_control, dtype=float)
        y_control = np.asarray(y_control, dtype=float)
        x_out = np.linspace(x_control[0], x_control[-1], int(num_points))
        return np.interp(x_out, x_control, y_control)

    def control_heights_and_node_values(self):
        """控制高度与各点初值；每点为 [T, t, fs, x, rhob, p]。"""
        return (
            [self.H0, self.H1, self.HH],
            [self.value0, self.value1, self.valueH],
        )

    def initial_bvp_guess(self, num_points=None):
        """
        Returns:
            y_guess: ndarray, shape (6, num_points)，行次序 T,t,fs,x,rhob,p
            H_ctrl: list of control heights
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


def create_standard_case_DOWN(case_type="default"):
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