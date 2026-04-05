"""
参考剖面初值的 BVP / HC 收敛性测试（与参考偏差不设阈值）。

求解流程与主程序一致：
  - BVP：`hegang.py` — `create_standard_case("initial_case")` 上半段 `FurnaceModel` +
    `create_standard_case_DOWN("initial_case_DOWN")` 下半段 `FurnaceModel_DOWN`，
    风口煤气迭代直至 |y_new - y_in| < 0.01 或达到最大次数。
    每次 `run()` 内部均调用 `params.initial_bvp_guess()`；本脚本通过替换该方法，
    使初值来自两段 CSV 拼接后的参考剖面（见下）。
  - HC：`hegang_hc.py` — 上半 `HCFurnaceModel.test_hc_4n4()` + 下半 `HCFurnaceModel_DOWN.test_hc_6()`，
    同样风口迭代；初值同样来自拼接参考剖面（替换 `initial_bvp_guess`）。
  - 耦合迭代中：每次 `run()` 须 `bvp_success==True`，每次 `test_hc_*` 须 `hc_converged==True`，
    否则立即判失败（`coupling_checks`）。

参考剖面构造
  - BVP：默认 `data/initial_case_bvp_0.0-4.2m_loop.csv` 与
    `data/initial_case_DOWN_bvp_4.2-5.9m_loop.csv` 按 z 排序拼接。
    对 z >= `--z-patch`（默认 4.2 m）的节点强制 w=0、y=1-x（与下半 CSV 缺列时一致）。
  - HC：默认 `data/test_hc_4n4_1e-3_UP_loop.csv` 与下半
    `data/test_hc_4n4_1e-3_DOWN_loop.csv`；若后者不存在则回退为
    `data/test_hc_6_1e-3_DOWN_loop.csv`（与 `hegang_hc.py` 中 `test_hc_6` 一致）。
    下半 CSV 无 y、w 列时补 y=1-x、w=0，并在 z >= `--z-patch` 上同样强制。

初值扩缩：`--scale` 为整体乘子；在插值到各段 `initial_mesh` 后对状态同乘（fs 等按行 clip）。

用法示例：
  python scripts/test_convergence_ref_profiles.py
  python scripts/test_convergence_ref_profiles.py --scale 0.95 1.0 1.05
  python scripts/test_convergence_ref_profiles.py --initial-mesh-up 100 --initial-mesh-down 50

日志 / 汇总：`--log-bvp`、`--log-hc` 在每次会话开头/结尾写入横幅（argv、scale 列表、网格、参考 CSV、合并行数、HC_REL_TOL_MAIN 等）；
每条试验记录 trial、scale、阶段、耗时、耦合结果与异常摘要；`--output-csv` 为汇总表。BVP/HC 在各自子目录下运行。
耦合 BVP/HC 结束后的剖面 CSV（原 run/test_hc 内 to_csv）由本脚本写入对应 `tmp/.../case_*` 目录。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (SRC, ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import matplotlib

matplotlib.use("Agg")

from furnace_model import FurnaceModel, HCFurnaceModel
from furnace_model_DOWN import FurnaceModel_DOWN, HCFurnaceModel_DOWN
from parameters import create_standard_case
from parameters_DOWN import create_standard_case_DOWN
from coupling_checks import require_bvp_segment_converged, require_hc_segment_converged
from paths import ensure_dirs

try:
    from hc_solver_settings import HC_REL_TOL_MAIN
except Exception:  # pragma: no cover
    HC_REL_TOL_MAIN = None

LOG = logging.getLogger("conv_ref")

# ---------------------------------------------------------------------------
# 默认参考文件（与 hegang / hegang_hc 及 data 目录一致）
# ---------------------------------------------------------------------------
DEFAULT_BVP_UP = ROOT / "data" / "bvp_0.0-4.2m_loop.csv"
DEFAULT_BVP_DOWN = ROOT / "data" / "bvp_4.2-5.9m_loop.csv"
DEFAULT_HC_UP = ROOT / "data" / "hc_4n4_0.0-4.2m_loop.csv"
DEFAULT_HC_DOWN_PRIMARY = ROOT / "data" / "hc_4n4_0.0-4.2m_loop.csv"
DEFAULT_HC_DOWN_FALLBACK = ROOT / "data" / "hc_6_4.2-5.9m_loop.csv"

DEFAULT_LOG_BVP = ROOT / "logs" / "convergence_ref_profiles_bvp.log"
DEFAULT_LOG_HC = ROOT / "logs" / "convergence_ref_profiles_hc.log"
DEFAULT_OUTPUT = ROOT / "output" / "convergence_ref_profiles_summary.csv"
TMP_RUN = ROOT / "tmp" / "convergence_ref_profiles_runs"

BVP_UP_ROWS = ("T", "t", "fs", "x", "y", "w", "rhob", "p")
BVP_DOWN_ROWS = ("T", "t", "fs", "x", "rhob", "p")
HC_UP_ROWS = BVP_UP_ROWS


def _configure_hybrid_logging(
    log_path: Path,
    *,
    append: bool = True,
    channel: str = "",
) -> None:
    """将 root 日志同时写入指定文件与 stderr；channel 用于区分 BVP/HC 文件中的前缀。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    tag = f"[conv_ref:{channel}] " if channel else "[conv_ref] "
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | " + tag + "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    mode = "a" if append else "w"
    fh = logging.FileHandler(log_path, encoding="utf-8", mode=mode)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    LOG.setLevel(logging.INFO)


def _append_log_banner(log_path: Path, lines: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _resolve_hc_down_csv(path: Path) -> Path:
    if path.is_file():
        return path
    if path == DEFAULT_HC_DOWN_PRIMARY and DEFAULT_HC_DOWN_FALLBACK.is_file():
        return DEFAULT_HC_DOWN_FALLBACK
    raise FileNotFoundError(
        f"HC 下半参考 CSV 不存在: {path}（若尚未生成 4n4 下半，可使用 {DEFAULT_HC_DOWN_FALLBACK}）"
    )


def merge_bvp_reference(
    path_up: Path,
    path_down: Path,
    *,
    z_patch: float,
) -> pd.DataFrame:
    """拼接上下 BVP 参考 CSV，z>=z_patch 处 w=0、y=1-x。"""
    df_u = pd.read_csv(path_up)
    df_d = pd.read_csv(path_down)
    for c in BVP_UP_ROWS:
        if c not in df_u.columns:
            raise ValueError(f"BVP 上半 CSV 缺少列 {c}: {path_up}")
    need_d = ["z", "T", "t", "fs", "x", "rhob", "p"]
    for c in need_d:
        if c not in df_d.columns:
            raise ValueError(f"BVP 下半 CSV 缺少列 {c}: {path_down}")
    df_d = df_d.copy()
    if "y" not in df_d.columns:
        df_d["y"] = 1.0 - df_d["x"].to_numpy(dtype=float)
    if "w" not in df_d.columns:
        df_d["w"] = 0.0
    cols = ["z"] + list(BVP_UP_ROWS)
    full = pd.concat([df_u[cols], df_d[cols]], ignore_index=True)
    full = full.sort_values("z").reset_index(drop=True)
    full = full.drop_duplicates(subset=["z"], keep="first")
    m = full["z"].to_numpy(dtype=float) >= float(z_patch)
    if np.any(m):
        xv = full.loc[m, "x"].to_numpy(dtype=float)
        full.loc[m, "y"] = 1.0 - xv
        full.loc[m, "w"] = 0.0
    return full


def merge_hc_reference(
    path_up: Path,
    path_down: Path,
    *,
    z_patch: float,
) -> pd.DataFrame:
    """拼接 HC 参考 CSV；下半无 y/w 时补全；z>=z_patch 强制 y=1-x、w=0。"""
    df_u = pd.read_csv(path_up)
    df_d = pd.read_csv(path_down)
    for c in HC_UP_ROWS:
        if c not in df_u.columns:
            raise ValueError(f"HC 上半 CSV 缺少列 {c}: {path_up}")
    need_d = ["z", "T", "t", "fs", "x", "rhob", "p"]
    for c in need_d:
        if c not in df_d.columns:
            raise ValueError(f"HC 下半 CSV 缺少列 {c}: {path_down}")
    df_d = df_d.copy()
    if "y" not in df_d.columns:
        df_d["y"] = 1.0 - df_d["x"].to_numpy(dtype=float)
    if "w" not in df_d.columns:
        df_d["w"] = 0.0
    cols = ["z"] + list(HC_UP_ROWS)
    full = pd.concat([df_u[cols], df_d[cols]], ignore_index=True)
    full = full.sort_values("z").reset_index(drop=True)
    full = full.drop_duplicates(subset=["z"], keep="first")
    m = full["z"].to_numpy(dtype=float) >= float(z_patch)
    if np.any(m):
        xv = full.loc[m, "x"].to_numpy(dtype=float)
        full.loc[m, "y"] = 1.0 - xv
        full.loc[m, "w"] = 0.0
    return full


def _interp_on_uniform(
    df: pd.DataFrame,
    z0: float,
    z1: float,
    n: int,
    keys: tuple[str, ...],
) -> np.ndarray:
    """在 [z0,z1] 上均匀 n 点，对 keys 列做线性插值；行为 len(keys)，列为 n。"""
    z_src = df["z"].to_numpy(dtype=float)
    order = np.argsort(z_src)
    z_src = z_src[order]
    z_tgt = np.linspace(float(z0), float(z1), int(n), dtype=float)
    rows: list[np.ndarray] = []
    for k in keys:
        v = df[k].to_numpy(dtype=float)[order]
        rows.append(np.interp(z_tgt, z_src, v).astype(float))
    return np.vstack(rows)


def apply_scale_bvp_up(y: np.ndarray, scale: float) -> np.ndarray:
    out = y * scale
    out[2] = np.clip(out[2], 0.0, 1.0)  # fs
    for i in (3, 4, 5):  # x, y, w
        out[i] = np.clip(out[i], 0.0, 1.0)
    return out.astype(float)


def apply_scale_bvp_down(y: np.ndarray, scale: float) -> np.ndarray:
    out = y * scale
    out[2] = np.clip(out[2], 0.0, 1.0)
    out[3] = np.clip(out[3], 0.0, 1.0)  # x
    return out.astype(float)


def bind_initial_bvp_guess_up(params, y_stack: np.ndarray) -> None:
    H_ctrl = [params.H0, params.HH]

    def initial_bvp_guess(num_points=None):  # noqa: ANN001
        if num_points is not None and int(num_points) != y_stack.shape[1]:
            z_old = np.linspace(params.H0, params.HH, y_stack.shape[1])
            z_new = np.linspace(params.H0, params.HH, int(num_points))
            rows = [np.interp(z_new, z_old, y_stack[i]) for i in range(8)]
            return np.vstack(rows), H_ctrl
        return y_stack.copy(), H_ctrl

    params.initial_bvp_guess = initial_bvp_guess  # type: ignore[method-assign]


def bind_initial_bvp_guess_down(params, y_stack: np.ndarray) -> None:
    H_ctrl = [params.H0, params.HH]

    def initial_bvp_guess(num_points=None):  # noqa: ANN001
        if num_points is not None and int(num_points) != y_stack.shape[1]:
            z_old = np.linspace(params.H0, params.HH, y_stack.shape[1])
            z_new = np.linspace(params.H0, params.HH, int(num_points))
            rows = [np.interp(z_new, z_old, y_stack[i]) for i in range(6)]
            return np.vstack(rows), H_ctrl
        return y_stack.copy(), H_ctrl

    params.initial_bvp_guess = initial_bvp_guess  # type: ignore[method-assign]


def _save_coupled_bvp_profiles(
    out_dir: Path | None,
    model_up,
    model_down,
    params_up,
    params_down,
) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_u = getattr(model_up, "last_bvp_profile_df", None)
    if df_u is not None:
        df_u.to_csv(
            out_dir / f"bvp_{params_up.H0:.1f}-{params_up.HH:.1f}m_loop.csv",
            index=False,
        )
    df_d = getattr(model_down, "last_bvp_profile_df", None)
    if df_d is not None:
        df_d.to_csv(
            out_dir / f"bvp_{params_down.H0:.1f}-{params_down.HH:.1f}m_loop.csv",
            index=False,
        )


def _save_coupled_hc_profiles(out_dir: Path | None, model_up, model_down) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_u = getattr(model_up, "last_hc_profile_df", None)
    if df_u is not None:
        df_u.to_csv(out_dir / "test_hc_4n4_1e-3_UP_loop_debug.csv", index=False)
    df_d = getattr(model_down, "last_hc_profile_df", None)
    if df_d is not None:
        df_d.to_csv(out_dir / "test_hc_6_1e-3_DOWN_loop_debug.csv", index=False)


def run_hegang_bvp_coupled(
    params_up,
    params_down,
    y_up: np.ndarray,
    y_down: np.ndarray,
    *,
    max_outer: int,
    y_tol: float,
    profile_out_dir: Path | None = None,
) -> tuple[bool, int, str | None]:
    """
    与 hegang.py 相同的风口耦合迭代。
    返回 (coupling_ok, outer_iterations, error_message)。
    coupling_ok：结束时 |y_new - y_in| <= y_tol 且无异常；且每次上半/下半 BVP 的 bvp_success 均为 True。
    profile_out_dir：若给定，在成功结束时将上下 BVP 剖面写入该目录。
    """
    bind_initial_bvp_guess_up(params_up, y_up)
    bind_initial_bvp_guess_down(params_down, y_down)
    model1 = FurnaceModel(params_up)
    err: str | None = None
    outer = 0
    try:
        results_UP = model1.run()
        require_bvp_segment_converged(results_UP, segment="up")
        t_up = results_UP["t_out"]
        fs_up = results_UP["fs_out"]
        rhob_up = results_UP["rhob_out"]
        p_up = results_UP["p_bottom"]
        params_down.t_in = t_up
        params_down.fs_in = fs_up
        params_down.rhob_in = rhob_up
        params_down.p0 = p_up
        params_down.p_in = p_up
        model2 = FurnaceModel_DOWN(params_down)
        results_DOWN = model2.run()
        require_bvp_segment_converged(results_DOWN, segment="down")
        T_down = results_DOWN["T_out"]
        x_down = results_DOWN["x_out"]
        F_b_DOWN = (
            2 * params_down.HI_O2 * params_down.Prod / 24
            + (1 - fs_up) * params_down.W_o / params_down.rho_po * params_down.c_H0 * 3 * 22.414
        )
        F_b_UP = F_b_DOWN + params_up.H2_input + params_up.CO_input
        T_new = (F_b_DOWN * T_down + (params_up.H2_input + params_up.CO_input) * 1223) / F_b_UP
        x_new = (F_b_DOWN * x_down + params_up.CO_input) / F_b_UP
        y_new = F_b_DOWN * (1 - x_down) / F_b_UP
        count = 0
        while (abs(y_new - params_up.y_in) > y_tol) and (count < max_outer):
            count += 1
            params_up.T_in = T_new
            params_up.x_in = x_new
            params_up.y_in = y_new
            results_UP = model1.run()
            require_bvp_segment_converged(results_UP, segment="up")
            t_up = results_UP["t_out"]
            fs_up = results_UP["fs_out"]
            rhob_up = results_UP["rhob_out"]
            p_up = results_UP["p_bottom"]
            params_down.t_in = t_up
            params_down.fs_in = fs_up
            params_down.rhob_in = rhob_up
            params_down.p0 = p_up
            params_down.p_in = p_up
            model2 = FurnaceModel_DOWN(params_down)
            results_DOWN = model2.run()
            require_bvp_segment_converged(results_DOWN, segment="down")
            T_down = results_DOWN["T_out"]
            x_down = results_DOWN["x_out"]
            F_b_DOWN = (
                2 * params_down.HI_O2 * params_down.Prod / 24
                + (1 - fs_up) * params_down.W_o / params_down.rho_po * params_down.c_H0 * 3 * 22.414
            )
            F_b_UP = F_b_DOWN + params_up.H2_input + params_up.CO_input
            T_new = (F_b_DOWN * T_down + (params_up.H2_input + params_up.CO_input) * 1223) / F_b_UP
            x_new = (F_b_DOWN * x_down + params_up.CO_input) / F_b_UP
            y_new = F_b_DOWN * (1 - x_down) / F_b_UP
        outer = count
        coupling_ok = abs(y_new - params_up.y_in) <= y_tol
        _save_coupled_bvp_profiles(profile_out_dir, model1, model2, params_up, params_down)
        return coupling_ok, outer, None
    except Exception as e:
        err = str(e)
        logging.exception("BVP 耦合运行异常: %s", e)
        return False, outer, err


def run_hegang_hc_coupled(
    params_up,
    params_down,
    y_up: np.ndarray,
    y_down: np.ndarray,
    *,
    max_outer: int,
    y_tol: float,
    profile_out_dir: Path | None = None,
) -> tuple[bool, int, str | None]:
    """与 hegang_hc.py 相同：test_hc_4n4 + test_hc_6 + 风口迭代；每次半段 hc_converged 须为 True。成功时写出 HC 剖面 CSV。"""
    bind_initial_bvp_guess_up(params_up, y_up)
    bind_initial_bvp_guess_down(params_down, y_down)
    model1 = HCFurnaceModel(params_up)
    outer = 0
    try:
        results_UP = model1.test_hc_4n4()
        require_hc_segment_converged(results_UP, segment="up")
        t_up = results_UP["t_out"]
        fs_up = results_UP["fs_out"]
        rhob_up = results_UP["rhob_out"]
        p_up = results_UP["p_bottom"]
        params_down.t_in = t_up
        params_down.fs_in = fs_up
        params_down.rhob_in = rhob_up
        params_down.p0 = p_up
        params_down.p_in = p_up
        model2 = HCFurnaceModel_DOWN(params_down)
        results_DOWN = model2.test_hc_6()
        require_hc_segment_converged(results_DOWN, segment="down")
        T_down = results_DOWN["T_out"]
        x_down = results_DOWN["x_out"]
        F_b_DOWN = (
            2 * params_down.HI_O2 * params_down.Prod / 24
            + (1 - fs_up) * params_down.W_o / params_down.rho_po * params_down.c_H0 * 3 * 22.414
        )
        F_b_UP = F_b_DOWN + params_up.H2_input + params_up.CO_input
        T_new = (F_b_DOWN * T_down + (params_up.H2_input + params_up.CO_input) * 1223) / F_b_UP
        x_new = (F_b_DOWN * x_down + params_up.CO_input) / F_b_UP
        y_new = F_b_DOWN * (1 - x_down) / F_b_UP
        count = 0
        while (abs(y_new - params_up.y_in) > y_tol) and (count < max_outer):
            count += 1
            params_up.T_in = T_new
            params_up.x_in = x_new
            params_up.y_in = y_new
            results_UP = model1.test_hc_4n4()
            require_hc_segment_converged(results_UP, segment="up")
            t_up = results_UP["t_out"]
            fs_up = results_UP["fs_out"]
            rhob_up = results_UP["rhob_out"]
            p_up = results_UP["p_bottom"]
            params_down.t_in = t_up
            params_down.fs_in = fs_up
            params_down.rhob_in = rhob_up
            params_down.p0 = p_up
            params_down.p_in = p_up
            model2 = HCFurnaceModel_DOWN(params_down)
            results_DOWN = model2.test_hc_6()
            require_hc_segment_converged(results_DOWN, segment="down")
            T_down = results_DOWN["T_out"]
            x_down = results_DOWN["x_out"]
            F_b_DOWN = (
                2 * params_down.HI_O2 * params_down.Prod / 24
                + (1 - fs_up) * params_down.W_o / params_down.rho_po * params_down.c_H0 * 3 * 22.414
            )
            F_b_UP = F_b_DOWN + params_up.H2_input + params_up.CO_input
            T_new = (F_b_DOWN * T_down + (params_up.H2_input + params_up.CO_input) * 1223) / F_b_UP
            x_new = (F_b_DOWN * x_down + params_up.CO_input) / F_b_UP
            y_new = F_b_DOWN * (1 - x_down) / F_b_UP
        outer = count
        coupling_ok = abs(y_new - params_up.y_in) <= y_tol
        _save_coupled_hc_profiles(profile_out_dir, model1, model2)
        return coupling_ok, outer, None
    except Exception as e:
        logging.exception("HC 耦合运行异常: %s", e)
        return False, outer, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="参考剖面初值 BVP/HC 收敛性测试（hegang / hegang_hc 流程）")
    parser.add_argument("--bvp-csv-up", type=Path, default=DEFAULT_BVP_UP, help="BVP 上半参考 CSV")
    parser.add_argument("--bvp-csv-down", type=Path, default=DEFAULT_BVP_DOWN, help="BVP 下半参考 CSV")
    parser.add_argument("--hc-csv-up", type=Path, default=DEFAULT_HC_UP, help="HC 上半参考 CSV")
    parser.add_argument("--hc-csv-down", type=Path, default=DEFAULT_HC_DOWN_PRIMARY, help="HC 下半参考 CSV")
    parser.add_argument(
        "--z-patch",
        type=float,
        default=4.2,
        help="z>=该高度 (m) 时令 w=0、y=1-x（拼接后剖面）",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[1.0],
        metavar="S",
        help="初值整体扩缩；多值扫参",
    )
    parser.add_argument("--initial-mesh-up", type=int, default=None, help="上半 initial_mesh（默认与算例一致）")
    parser.add_argument("--initial-mesh-down", type=int, default=None, help="下半 initial_mesh（默认与算例一致）")
    parser.add_argument("--U", type=float, default=None, help="覆盖上半 FurnaceParameters.U（可选）")
    parser.add_argument("--log-bvp", type=Path, default=DEFAULT_LOG_BVP, help="BVP 日志（追加）")
    parser.add_argument("--log-hc", type=Path, default=DEFAULT_LOG_HC, help="HC 日志（追加）")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT, help="汇总 CSV")
    parser.add_argument("--case-bvp", type=str, default="convref_bvp", help="BVP 子目录名前缀")
    parser.add_argument("--case-hc", type=str, default="convref_hc", help="HC 子目录名前缀")
    parser.add_argument("--max-outer", type=int, default=100, help="风口 y 耦合最大迭代次数（与 hegang 一致）")
    parser.add_argument("--y-tol", type=float, default=0.01, help="|y_new - y_in| 收敛阈值")
    parser.add_argument(
        "--bvp-verbose",
        type=int,
        default=0,
        help="保留参数；当前 FurnaceModel.solve_bvp 内部 verbose 由源码固定，此参数仅占位",
    )
    args = parser.parse_args()
    _ = args.bvp_verbose

    ensure_dirs()
    TMP_RUN.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    for pth, label in (
        (args.bvp_csv_up, "BVP 上半"),
        (args.bvp_csv_down, "BVP 下半"),
        (args.hc_csv_up, "HC 上半"),
    ):
        if not pth.is_file():
            raise FileNotFoundError(f"{label} CSV 不存在: {pth}")

    hc_down_resolved = _resolve_hc_down_csv(args.hc_csv_down)

    df_bvp = merge_bvp_reference(args.bvp_csv_up, args.bvp_csv_down, z_patch=args.z_patch)
    df_hc = merge_hc_reference(args.hc_csv_up, hc_down_resolved, z_patch=args.z_patch)

    scale_list = [float(s) for s in args.scale]
    cwd = os.getcwd()
    results_rows: list[dict] = []

    _pu = create_standard_case("initial_case")
    _pd = create_standard_case_DOWN("initial_case_DOWN")
    _mesh_u = int(args.initial_mesh_up) if args.initial_mesh_up is not None else int(_pu.initial_mesh)
    _mesh_d = int(args.initial_mesh_down) if args.initial_mesh_down is not None else int(_pd.initial_mesh)
    _banner_common = [
        "",
        "=" * 72,
        f"convergence_ref_profiles | cwd={cwd}",
        f"argv={' '.join(sys.argv)}",
        f"scales ({len(scale_list)})={scale_list}",
        f"z_patch_m={args.z_patch} | max_outer={args.max_outer} | y_tol={args.y_tol}",
        f"initial_mesh_up={_mesh_u} initial_mesh_down={_mesh_d} | U_override={args.U}",
        f"BVP CSV up={args.bvp_csv_up} down={args.bvp_csv_down}",
        f"HC  CSV up={args.hc_csv_up} down_resolved={hc_down_resolved}",
        f"merged BVP rows={len(df_bvp)} z=[{df_bvp['z'].min():.4f},{df_bvp['z'].max():.4f}]",
        f"merged HC  rows={len(df_hc)} z=[{df_hc['z'].min():.4f},{df_hc['z'].max():.4f}]",
        f"TMP_RUN={TMP_RUN.resolve()} | output_csv={args.output_csv.resolve()}",
    ]
    if HC_REL_TOL_MAIN is not None:
        _banner_common.append(
            f"HC_REL_TOL_MAIN={HC_REL_TOL_MAIN} (test_hc_* per-variable relative tolerance)",
        )
    _banner_common.append("=" * 72)
    for _lp in (args.log_bvp, args.log_hc):
        _append_log_banner(_lp, _banner_common)

    t_session0 = time.perf_counter()

    for i, scale in enumerate(scale_list):
        case_bvp = f"{args.case_bvp}_i{i}"
        case_hc = f"{args.case_hc}_i{i}"
        run_bvp_dir = TMP_RUN / case_bvp
        run_hc_dir = TMP_RUN / case_hc
        run_bvp_dir.mkdir(parents=True, exist_ok=True)
        run_hc_dir.mkdir(parents=True, exist_ok=True)

        params_up_b = create_standard_case("initial_case")
        params_down_b = create_standard_case_DOWN("initial_case_DOWN")
        if args.initial_mesh_up is not None:
            params_up_b.initial_mesh = int(args.initial_mesh_up)
        if args.initial_mesh_down is not None:
            params_down_b.initial_mesh = int(args.initial_mesh_down)
        if args.U is not None:
            params_up_b.U = float(args.U)

        params_up_h = create_standard_case("initial_case")
        params_down_h = create_standard_case_DOWN("initial_case_DOWN")
        if args.initial_mesh_up is not None:
            params_up_h.initial_mesh = int(args.initial_mesh_up)
        if args.initial_mesh_down is not None:
            params_down_h.initial_mesh = int(args.initial_mesh_down)
        if args.U is not None:
            params_up_h.U = float(args.U)

        y_bvp_u = _interp_on_uniform(
            df_bvp,
            params_up_b.H0,
            params_up_b.HH,
            params_up_b.initial_mesh,
            BVP_UP_ROWS,
        )
        y_bvp_d = _interp_on_uniform(
            df_bvp,
            params_down_b.H0,
            params_down_b.HH,
            params_down_b.initial_mesh,
            BVP_DOWN_ROWS,
        )
        y_bvp_u = apply_scale_bvp_up(y_bvp_u, scale)
        y_bvp_d = apply_scale_bvp_down(y_bvp_d, scale)

        y_hc_u = _interp_on_uniform(
            df_hc,
            params_up_h.H0,
            params_up_h.HH,
            params_up_h.initial_mesh,
            HC_UP_ROWS,
        )
        y_hc_d = _interp_on_uniform(
            df_hc,
            params_down_h.H0,
            params_down_h.HH,
            params_down_h.initial_mesh,
            BVP_DOWN_ROWS,
        )
        y_hc_u = apply_scale_bvp_up(y_hc_u, scale)
        y_hc_d = apply_scale_bvp_down(y_hc_d, scale)

        row: dict = {
            "scale_idx": i,
            "bvp_csv_up": str(args.bvp_csv_up),
            "bvp_csv_down": str(args.bvp_csv_down),
            "hc_csv_up": str(args.hc_csv_up),
            "hc_csv_down": str(hc_down_resolved),
            "z_patch_m": args.z_patch,
            "scale": scale,
            "initial_mesh_up": params_up_b.initial_mesh,
            "initial_mesh_down": params_down_b.initial_mesh,
            "U": params_up_b.U,
            "case_bvp": case_bvp,
            "case_hc": case_hc,
        }

        # ---------- BVP ----------
        _configure_hybrid_logging(args.log_bvp, channel="BVP")
        LOG.info(
            "trial %d/%d | scale=%s | BVP phase | mesh_up=%s mesh_down=%s | dir=%s",
            i + 1,
            len(scale_list),
            scale,
            params_up_b.initial_mesh,
            params_down_b.initial_mesh,
            run_bvp_dir.resolve(),
        )
        t0 = time.perf_counter()
        bvp_exc: str | None = None
        bvp_ok = False
        bvp_outer = 0
        try:
            os.chdir(run_bvp_dir)
            bvp_ok, bvp_outer, bvp_exc = run_hegang_bvp_coupled(
                params_up_b,
                params_down_b,
                y_bvp_u,
                y_bvp_d,
                max_outer=args.max_outer,
                y_tol=args.y_tol,
                profile_out_dir=run_bvp_dir,
            )
        except Exception as e:
            bvp_exc = str(e)
            logging.exception("BVP 未捕获异常: %s", e)
        finally:
            os.chdir(cwd)

        elapsed_bvp = time.perf_counter() - t0
        row["bvp_coupling_ok"] = bvp_ok
        row["bvp_outer_iters"] = bvp_outer
        row["bvp_elapsed_s"] = elapsed_bvp
        row["bvp_log"] = str(args.log_bvp)
        row["bvp_run_dir"] = str(run_bvp_dir)
        row["bvp_exception"] = bvp_exc or ""
        row["bvp_final_success"] = bvp_ok and not bvp_exc
        LOG.info(
            "trial %d/%d | scale=%s | BVP end | coupling_ok=%s outer_iters=%d elapsed=%.2fs | final_success=%s",
            i + 1,
            len(scale_list),
            scale,
            bvp_ok,
            bvp_outer,
            elapsed_bvp,
            row["bvp_final_success"],
        )
        if bvp_exc:
            _ex = bvp_exc if len(bvp_exc) <= 500 else bvp_exc[:500] + "…"
            LOG.warning("trial %d/%d | scale=%s | BVP exception | %s", i + 1, len(scale_list), scale, _ex)

        # ---------- HC ----------
        _configure_hybrid_logging(args.log_hc, channel="HC")
        LOG.info(
            "trial %d/%d | scale=%s | HC phase | mesh_up=%s mesh_down=%s | dir=%s",
            i + 1,
            len(scale_list),
            scale,
            params_up_h.initial_mesh,
            params_down_h.initial_mesh,
            run_hc_dir.resolve(),
        )
        t1 = time.perf_counter()
        hc_exc: str | None = None
        hc_ok = False
        hc_outer = 0
        try:
            os.chdir(run_hc_dir)
            hc_ok, hc_outer, hc_exc = run_hegang_hc_coupled(
                params_up_h,
                params_down_h,
                y_hc_u,
                y_hc_d,
                max_outer=args.max_outer,
                y_tol=args.y_tol,
                profile_out_dir=run_hc_dir,
            )
        except Exception as e:
            hc_exc = str(e)
            logging.exception("HC 未捕获异常: %s", e)
        finally:
            os.chdir(cwd)

        elapsed_hc = time.perf_counter() - t1
        row["hc_coupling_ok"] = hc_ok
        row["hc_outer_iters"] = hc_outer
        row["hc_elapsed_s"] = elapsed_hc
        row["hc_log"] = str(args.log_hc)
        row["hc_run_dir"] = str(run_hc_dir)
        row["hc_exception"] = hc_exc or ""
        row["hc_converged"] = hc_ok and not hc_exc
        LOG.info(
            "trial %d/%d | scale=%s | HC end | coupling_ok=%s outer_iters=%d elapsed=%.2fs | final_success=%s",
            i + 1,
            len(scale_list),
            scale,
            hc_ok,
            hc_outer,
            elapsed_hc,
            row["hc_converged"],
        )
        if hc_exc:
            _ex = hc_exc if len(hc_exc) <= 500 else hc_exc[:500] + "…"
            LOG.warning("trial %d/%d | scale=%s | HC exception | %s", i + 1, len(scale_list), scale, _ex)

        overall = bool(row["bvp_final_success"] or row["hc_converged"])
        row["overall_success"] = overall
        row["note"] = (
            "BVP/HC 成功：每段 BVP bvp_success、每段 HC hc_converged 为真，"
            "风口 y 耦合满足阈值且无异常；剖面与参考的 RMSE 未判定"
        )
        results_rows.append(row)

        print(
            f"[{i+1}/{len(scale_list)}] scale={scale}\n"
            f"  BVP: coupling_ok={bvp_ok}, outer={bvp_outer}, {elapsed_bvp:.2f}s, success={row['bvp_final_success']}\n"
            f"  HC: coupling_ok={hc_ok}, outer={hc_outer}, {elapsed_hc:.2f}s, success={row['hc_converged']}\n"
            f"  overall_success={overall}"
        )

    pd.DataFrame(results_rows).to_csv(args.output_csv, index=False)
    _wall = time.perf_counter() - t_session0
    _bvp_n = sum(1 for r in results_rows if r.get("bvp_final_success"))
    _hc_n = sum(1 for r in results_rows if r.get("hc_converged"))
    _ov_n = sum(1 for r in results_rows if r.get("overall_success"))
    _summary_tail = [
        "",
        "=" * 72,
        f"convergence_ref_profiles | session_end | wall_s={_wall:.1f}",
        f"trials={len(results_rows)} | BVP_ok={_bvp_n} HC_ok={_hc_n} overall_ok={_ov_n}",
        f"summary_csv={args.output_csv.resolve()}",
        "=" * 72,
    ]
    for _lp in (args.log_bvp, args.log_hc):
        _append_log_banner(_lp, _summary_tail)
    print(
        f"汇总已写入: {args.output_csv} （共 {len(results_rows)} 行）| "
        f"BVP_ok={_bvp_n} HC_ok={_hc_n} overall_ok={_ov_n} | wall={_wall:.1f}s"
    )


if __name__ == "__main__":
    main()
