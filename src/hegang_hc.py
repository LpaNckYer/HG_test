import logging
from rizhi import setup_logging

from coupling_checks import require_hc_segment_converged
from parameters import create_standard_case
from parameters_DOWN import create_standard_case_DOWN
from furnace_model import HCFurnaceModel
from furnace_model_DOWN import HCFurnaceModel_DOWN


logger = setup_logging('loop.log')

logger.info("热量流法程序开始运行")

try:

    params_UP = create_standard_case("initial_case")
    model1 = HCFurnaceModel(params_UP)
    logging.info("求解上半部分模型-hc")
    results_UP = model1.test_hc_4n4()
    require_hc_segment_converged(results_UP, segment="up")

    t_up = results_UP['t_out']
    fs_up = results_UP['fs_out']
    rhob_up = results_UP['rhob_out']
    p_up = results_UP['p_bottom']

    params_DOWN = create_standard_case_DOWN("initial_case_DOWN")

    params_DOWN.t_in = t_up
    params_DOWN.fs_in = fs_up
    params_DOWN.rhob_in = rhob_up
    params_DOWN.p0 = p_up
    params_DOWN.p_in = p_up
    model2 = HCFurnaceModel_DOWN(params_DOWN)
    logging.info("求解下半部分模型-hc")
    results_DOWN = model2.test_hc_6()
    require_hc_segment_converged(results_DOWN, segment="down")

    T_down = results_DOWN['T_out']
    x_down = results_DOWN['x_out']

    F_b_DOWN = 2 * params_DOWN.HI_O2 * params_DOWN.Prod / 24 + (1-fs_up)*params_DOWN.W_o/params_DOWN.rho_po*params_DOWN.c_H0*3*22.414 # [Nm3/hr]
    F_b_UP = F_b_DOWN + params_UP.H2_input + params_UP.CO_input
    T_new = (F_b_DOWN*T_down + (params_UP.H2_input + params_UP.CO_input) * 1223) / F_b_UP # 混合气温度（忽略气体比热容差异）
    x_new = (F_b_DOWN*x_down + params_UP.CO_input) / F_b_UP
    y_new = F_b_DOWN*(1-x_down) / F_b_UP

    count = 0
    while (abs(y_new - params_UP.y_in) > 0.01) and (count < 100):
        count += 1
        print(f"第{count}次迭代，T_new={T_new:.2f}，T_in={params_UP.T_in:.2f}，x_new={x_new:.2f}，x_in={params_UP.x_in:.2f}，y_new={y_new:.2f}，y_in={params_UP.y_in:.2f}")

        params_UP.T_in = T_new
        params_UP.x_in = x_new
        params_UP.y_in = y_new
        logging.info("求解上半部分模型-hc")
        results_UP = model1.test_hc_4n4()
        require_hc_segment_converged(results_UP, segment="up")

        t_up = results_UP['t_out']
        fs_up = results_UP['fs_out']
        rhob_up = results_UP['rhob_out']
        p_up = results_UP['p_bottom']
   
        params_DOWN.t_in = t_up
        params_DOWN.fs_in = fs_up
        params_DOWN.rhob_in = rhob_up
        params_DOWN.p0 = p_up
        params_DOWN.p_in = p_up
        logging.info("求解下半部分模型-hc")
        results_DOWN = model2.test_hc_6()
        require_hc_segment_converged(results_DOWN, segment="down")

        T_down = results_DOWN['T_out']
        x_down = results_DOWN['x_out']

        F_b_DOWN = 2 * params_DOWN.HI_O2 * params_DOWN.Prod / 24 + (1-fs_up)*params_DOWN.W_o/params_DOWN.rho_po*params_DOWN.c_H0*3*22.414 # [Nm3/hr]
        F_b_UP = F_b_DOWN + params_UP.H2_input + params_UP.CO_input
        T_new = (F_b_DOWN*T_down + (params_UP.H2_input + params_UP.CO_input) * 1223) / F_b_UP # 混合气温度（忽略气体比热容差异）
        x_new = (F_b_DOWN*x_down + params_UP.CO_input) / F_b_UP
        y_new = F_b_DOWN*(1-x_down) / F_b_UP
    
    print(f"共{count}次迭代，T_new={T_new:.2f}，T_in={params_UP.T_in:.2f}，x_new={x_new:.2f}，x_in={params_UP.x_in:.2f}，y_new={y_new:.2f}，y_in={params_UP.y_in:.2f}")

except Exception as e:
    logger.error(f"程序出错: {e}", exc_info=True)

finally:
    logger.info("程序结束")