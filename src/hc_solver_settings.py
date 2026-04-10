"""热量流（HC）子求解与 test_hc_* 耦合迭代的容差与迭代上限（单点维护）。"""

# —— 相对误差阈值 ||Δu|| / ||u||（与 numpy.linalg.norm 一致）——
HC_REL_TOL_MAIN = 1e-3   # T,t,x,y 及 p,rhob,w,fs 等在耦合/测试中的主判据
HC_REL_TOL_TIGHT = 1e-4  # w_hc、fs_hc 内部 Picard 迭代（更严）

# —— 单个子程序内松弛迭代（Tt_hc / xy_hc / w_hc / fs_hc）——
HC_MAX_ITER_TT_XY_FS = 50
HC_MAX_ITER_W = 100
HC_RELAXATION = 0.5

# —— test_hc_* 等多层耦合 ——
HC_MAX_ITER_TEST_NESTED_INNER = 100   # 嵌套内层 count_in
HC_MAX_ITER_TEST_OUTER_UP = 100       # 上半部 count_out / 全变量单循环
HC_MAX_ITER_TEST_OUTER_DOWN_DEEP = 100  # 下半部 test_hc_3n3 外层
HC_MAX_ITER_TEST_MONOLITHIC = 1000    # test_hc_6 等单一大循环

# 首轮 T,t,x(,y) 耦合（原下半 test 首轮为 10，上半为 100，在此分别命名）
HC_MAX_ITER_TEST_FIRST_UP = 100
HC_MAX_ITER_TEST_FIRST_DOWN = 10
