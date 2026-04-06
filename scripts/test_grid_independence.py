"""
网格无关性测试：上半（furnace_model）与下半（furnace_model_DOWN），分别用 BVP 与 HC 求解。

- region=up + mode=bvp：FurnaceModel.run()，扫描 initial_mesh
- region=up + mode=hc：HCFurnaceModel.test_hc_4n4()（与 hegang_hc 上半一致）
- region=down + mode=bvp：FurnaceModel_DOWN.run()
- region=down + mode=hc：HCFurnaceModel_DOWN.test_hc_6()（与 hegang_hc 下半一致）

方程收敛：
- BVP：无残差字段时，以本次运行未抛异常（status=success）为准；若 results 中含 bvp_* 则仍按阈值判据
- HC：需 status=success 且 hc_converged 为 True（test_hc_* 末尾各相对误差均 < HC_REL_TOL_MAIN）；触达最大迭代轮次时 hc_converged 为 False

出口稳定判据：与参考网格（首个满足方程收敛的行）比较相对/绝对误差；参与比较的出口量随 region 变化（下半无 y/w/fl）。

用法示例：
  python scripts/test_grid_independence.py --region up --mode bvp
  python scripts/test_grid_independence.py --region up --mode hc --hc-case default_case
  python scripts/test_grid_independence.py --region down --mode bvp
  python scripts/test_grid_independence.py --region down --mode hc --down-case initial_case_DOWN
  python scripts/test_grid_independence.py --region down --mode hc --meshes 300,200,100,50

进程日志默认 logs/grid_independence_<region>_<mode>.log（时间戳 + level + [grid_independence] 前缀；
含会话 argv、判据阈值、每网格 begin/end、equation_converged、失败摘要与总耗时 wall_s）。
汇总 CSV 默认 output/grid_independence_<region>_<mode>.csv，**仅含列**：
region, solver, initial_mesh, bvp_final_nodes, bvp_success, hc_final_max_re, hc_success,
T_out, t_out, fs_out, x_out, y_out, w_out, rhob_out, p_bottom, solution_stable_vs_ref
（未参与求解的字段为空；与参考网格比较后的稳定性仅在最终行填入）。
编码为 **utf-8-sig**，便于 Excel 正确显示中文。

BVP/HC 剖面 CSV（原 furnace_model*.run / test_hc_* 内 to_csv）改在本脚本每次成功求解后写入 --tmp 对应子目录
（与原先 chdir 工作目录行为一致）；文件名带 `mesh{initial_mesh}` 以区分各轮网格。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
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
from parameters import FurnaceParameters, quick_modify
from parameters_DOWN import create_standard_case_DOWN
from paths import ensure_dirs, logs_path, output_path
from save_load import load_parameters

try:
    from hc_solver_settings import HC_REL_TOL_MAIN
except Exception:  # pragma: no cover
    HC_REL_TOL_MAIN = None

# 未指定 --meshes 时使用
DEFAULT_MESHES = [400, 300, 200, 150, 100, 75, 50, 30, 20, 10]

_CSV_ENCODING = "utf-8-sig"

GRID_INDEPENDENCE_SUMMARY_COLUMNS = (
    "region",
    "solver",
    "initial_mesh",
    "bvp_final_nodes",
    "bvp_success",
    "hc_final_max_re",
    "hc_success",
    "T_out",
    "t_out",
    "fs_out",
    "x_out",
    "y_out",
    "w_out",
    "rhob_out",
    "p_bottom",
    "solution_stable_vs_ref",
)

LOG = logging.getLogger("grid_independence")


def grid_row_to_summary(row: dict, *, solution_stable: bool | str | None = None) -> dict:
    """将内部宽表行压成写入 CSV 的列集（与 GRID_INDEPENDENCE_SUMMARY_COLUMNS 一致）。"""
    sol = row.get("solver")
    if solution_stable is None:
        stable_cell: bool | str = ""
    elif isinstance(solution_stable, str):
        stable_cell = solution_stable
    else:
        stable_cell = bool(solution_stable)
    hc_ok = (
        row.get("status") == "success" and bool(row.get("hc_outer_converged"))
        if sol == "hc"
        else ""
    )
    return {
        "region": row.get("region"),
        "solver": sol,
        "initial_mesh": row.get("initial_mesh"),
        "bvp_final_nodes": row.get("bvp_n_nodes_final") if sol == "bvp" else "",
        "bvp_success": row.get("bvp_success") if sol == "bvp" else "",
        "hc_final_max_re": row.get("hc_max_re_final") if sol == "hc" else "",
        "hc_success": hc_ok,
        "T_out": row.get("T_out"),
        "t_out": row.get("t_out"),
        "fs_out": row.get("fs_out"),
        "x_out": row.get("x_out"),
        "y_out": row.get("y_out", ""),
        "w_out": row.get("w_out", ""),
        "rhob_out": row.get("rhob_out"),
        "p_bottom": row.get("p_bottom"),
        "solution_stable_vs_ref": stable_cell,
    }


def write_grid_independence_summary_csv(path: Path, rows: list[dict]) -> None:
    recs = [grid_row_to_summary(r, solution_stable=r.get("solution_stable_vs_ref", "")) for r in rows]
    pd.DataFrame(recs, columns=list(GRID_INDEPENDENCE_SUMMARY_COLUMNS)).to_csv(
        path,
        index=False,
        encoding=_CSV_ENCODING,
    )


def write_grid_independence_progress_csv(path: Path, rows: list[dict]) -> None:
    """网格扫描过程中写入；尚未相对参考网格判稳时 solution_stable_vs_ref 为空。"""
    recs = [grid_row_to_summary(r, solution_stable="") for r in rows]
    pd.DataFrame(recs, columns=list(GRID_INDEPENDENCE_SUMMARY_COLUMNS)).to_csv(
        path,
        index=False,
        encoding=_CSV_ENCODING,
    )


def _fmt_outlet_log(x, nd: int = 5) -> str:
    """日志用出口数值：有限浮点保留 nd 位小数，否则原样字符串。"""
    if x is None:
        return "None"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return str(x)
    return f"{xf:.{nd}f}"


def _save_bvp_profile_if_any(model, workdir: Path, *, mesh: int) -> None:
    df = getattr(model, "last_bvp_profile_df", None)
    if df is None:
        return
    H0, HH = model.params.H0, model.params.HH
    workdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        workdir / f"bvp_{H0:.1f}-{HH:.1f}m_mesh{int(mesh)}_loop.csv",
        index=False,
        encoding=_CSV_ENCODING,
    )


def _save_hc_profile_if_any(model, workdir: Path, *, region: str, mesh: int) -> None:
    df = getattr(model, "last_hc_profile_df", None)
    if df is None:
        return
    name = (
        f"test_hc_4n4_1e-3_UP_mesh{int(mesh)}_loop_debug.csv"
        if region == "up"
        else f"test_hc_6_1e-3_DOWN_mesh{int(mesh)}_loop_debug.csv"
    )
    workdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(workdir / name, index=False, encoding=_CSV_ENCODING)


def outlet_keys(region: str) -> list[str]:
    if region == "down":
        return ["T_out", "t_out", "fs_out", "x_out", "rhob_out", "p_bottom"]
    return ["T_out", "t_out", "fs_out", "x_out", "y_out", "w_out", "rhob_out", "p_bottom"]


def fraction_keys_for_stability(region: str) -> list[str]:
    """用于绝对误差阈值的分数类出口（温度等仍看 rel_diff_outlet_max）。"""
    if region == "down":
        return ["fs_out", "x_out"]
    return ["fs_out", "x_out", "y_out", "w_out"]


@dataclass(frozen=True)
class Criteria:
    require_bvp_success: bool = True
    max_rms_le: float = 1e-3
    bc_l2_le: float = 1e-6
    rel_le: float = 1e-3
    abs_le_fractions: float = 3e-4


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def outlet_error_metrics(row: dict, ref: dict, keys: list[str]) -> dict[str, float]:
    tiny = 1e-12
    out: dict[str, float] = {}
    abs_diffs = []
    rel_diffs = []
    for k in keys:
        v = _safe_float(row.get(k))
        r = _safe_float(ref.get(k))
        ad = abs(v - r)
        rd = ad / max(abs(r), tiny)
        out[f"abs_diff_{k}"] = ad
        out[f"rel_diff_{k}"] = rd
        abs_diffs.append(ad)
        rel_diffs.append(rd)
    out["abs_diff_outlet_max"] = float(np.nanmax(abs_diffs)) if abs_diffs else float("nan")
    out["rel_diff_outlet_max"] = float(np.nanmax(rel_diffs)) if rel_diffs else float("nan")
    return out


def is_equation_converged(row: dict, c: Criteria) -> bool:
    if row.get("solver") == "hc":
        return row.get("status") == "success" and bool(row.get("hc_outer_converged"))

    if row.get("solver") != "bvp":
        return False

    if row.get("status") != "success":
        return False

    bs = row.get("bvp_success")
    if c.require_bvp_success and bs is False:
        return False

    max_rms = row.get("bvp_max_rms_residual_final")
    bc_l2 = row.get("bvp_bc_l2_residual_final")
    if max_rms is None and bc_l2 is None:
        return True
    if max_rms is not None:
        max_rms = _safe_float(max_rms)
        if not (np.isfinite(max_rms) and max_rms <= c.max_rms_le):
            return False
    if bc_l2 is not None:
        bc_l2 = _safe_float(bc_l2)
        if not (np.isfinite(bc_l2) and bc_l2 <= c.bc_l2_le):
            return False
    return True


def is_solution_stable(row: dict, c: Criteria, region: str) -> bool:
    if not np.isfinite(_safe_float(row.get("rel_diff_outlet_max"))):
        return False
    for k in fraction_keys_for_stability(region):
        ad = _safe_float(row.get(f"abs_diff_{k}"))
        if not (np.isfinite(ad) and ad <= c.abs_le_fractions):
            return False
    if _safe_float(row.get("rel_diff_outlet_max")) > c.rel_le:
        return False
    return True


def _results_row_bvp(results: dict, mesh: int, *, region: str) -> dict:
    keys = outlet_keys(region)
    row = {
        "region": region,
        "solver": "bvp",
        "initial_mesh": mesh,
        "hc_outer_converged": None,
        "hc_max_re_final": None,
        **{k: results.get(k) for k in (["bvp_success", "bvp_tol_final", "bvp_n_nodes_final", "bvp_max_rms_residual_final", "bvp_bc_l2_residual_final"] + keys)},
    }
    return row


def run_one_bvp_up(mesh: int, base: FurnaceParameters, workdir: Path, bvp_verbose: int = 0) -> dict:
    p = deepcopy(base)
    p.initial_mesh = int(mesh)
    p.case_name = f"grid_up_{mesh}"

    model = FurnaceModel(p)
    model.bvp_verbose = bvp_verbose

    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        t0 = time.perf_counter()
        status = "success"
        try:
            results = model.run()
        except Exception as e:
            status = f"fail: {e}"
            results = getattr(model, "results", {}) or {}
        elapsed = time.perf_counter() - t0
    finally:
        os.chdir(cwd)

    if status == "success":
        _save_bvp_profile_if_any(model, workdir, mesh=mesh)

    row = {
        "status": status,
        "elapsed_s": elapsed,
        **_results_row_bvp(results, mesh, region="up"),
    }
    return row


def run_one_bvp_down(mesh: int, base, workdir: Path) -> dict:
    p = quick_modify(base, case_name=f"grid_down_{mesh}", initial_mesh=int(mesh))

    model = FurnaceModel_DOWN(p)

    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        t0 = time.perf_counter()
        status = "success"
        try:
            results = model.run()
        except Exception as e:
            status = f"fail: {e}"
            results = getattr(model, "results", {}) or {}
        elapsed = time.perf_counter() - t0
    finally:
        os.chdir(cwd)

    if status == "success":
        _save_bvp_profile_if_any(model, workdir, mesh=mesh)

    row = {
        "status": status,
        "elapsed_s": elapsed,
        **_results_row_bvp(results, mesh, region="down"),
    }
    return row


def run_one_hc_up(mesh: int, hc_case: str | None, workdir: Path) -> dict:
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        t0 = time.perf_counter()
        status = "success"
        hc_outer_converged = False
        results: dict = {}
        model: HCFurnaceModel | None = None
        try:
            if hc_case:
                params = load_parameters(hc_case)
            else:
                params = FurnaceParameters()
                params.U = 10.0
            params2 = quick_modify(
                params,
                case_name=f"{getattr(params, 'case_name', 'hc')}_grid_{mesh}",
                initial_mesh=int(mesh),
            )
            model = HCFurnaceModel(params2)
            results = model.test_hc_4n4()
            hc_outer_converged = bool(results.get("hc_converged"))
        except Exception as e:
            status = f"fail: {e}"
            results = getattr(model, "results", {}) or {} if model is not None else {}
        elapsed = time.perf_counter() - t0
    finally:
        os.chdir(cwd)

    if status == "success" and model is not None:
        _save_hc_profile_if_any(model, workdir, region="up", mesh=mesh)

    keys = outlet_keys("up")
    return {
        "region": "up",
        "solver": "hc",
        "initial_mesh": mesh,
        "status": status,
        "elapsed_s": elapsed,
        "hc_outer_converged": hc_outer_converged,
        "hc_max_re_final": results.get("hc_max_re_final"),
        "bvp_success": None,
        "bvp_tol_final": None,
        "bvp_n_nodes_final": None,
        "bvp_max_rms_residual_final": None,
        "bvp_bc_l2_residual_final": None,
        **{k: results.get(k) for k in keys},
    }


def run_one_hc_down(mesh: int, down_case: str, workdir: Path) -> dict:
    base = create_standard_case_DOWN(down_case)
    p = quick_modify(base, case_name=f"grid_down_{mesh}", initial_mesh=int(mesh))

    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        t0 = time.perf_counter()
        status = "success"
        hc_outer_converged = False
        results: dict = {}
        model: HCFurnaceModel_DOWN | None = None
        try:
            model = HCFurnaceModel_DOWN(p)
            results = model.test_hc_6()
            hc_outer_converged = bool(results.get("hc_converged"))
        except Exception as e:
            status = f"fail: {e}"
            results = getattr(model, "results", {}) or {} if model is not None else {}
        elapsed = time.perf_counter() - t0
    finally:
        os.chdir(cwd)

    if status == "success" and model is not None:
        _save_hc_profile_if_any(model, workdir, region="down", mesh=mesh)

    keys = outlet_keys("down")
    return {
        "region": "down",
        "solver": "hc",
        "initial_mesh": mesh,
        "status": status,
        "elapsed_s": elapsed,
        "hc_outer_converged": hc_outer_converged,
        "hc_max_re_final": results.get("hc_max_re_final"),
        "bvp_success": None,
        "bvp_tol_final": None,
        "bvp_n_nodes_final": None,
        "bvp_max_rms_residual_final": None,
        "bvp_bc_l2_residual_final": None,
        **{k: results.get(k) for k in keys},
    }


def parse_meshes(s: str | None) -> list[int]:
    if not s or not str(s).strip():
        return list(DEFAULT_MESHES)
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def configure_progress_logging(log_file: Path, *, console: bool = True) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    datefmt = "%Y-%m-%d %H:%M:%S"
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [grid_independence] %(message)s",
        datefmt=datefmt,
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    LOG.setLevel(logging.INFO)


def _normalize_mode(s: str) -> str:
    if s == "hc_5n4":
        return "hc"
    return s


def main():
    parser = argparse.ArgumentParser(
        description="网格无关性：上半/下半 furnace 模型 × BVP 或 HC",
    )
    parser.add_argument("--region", choices=["up", "down"], default="up", help="up=furnace_model，down=furnace_model_DOWN")
    parser.add_argument(
        "--mode",
        choices=["bvp", "hc", "hc_5n4"],
        default="bvp",
        help="hc_5n4 与 hc 等价（兼容旧参数）",
    )
    parser.add_argument(
        "--meshes",
        default=None,
        help=f"逗号分隔的 initial_mesh 列表；默认 {','.join(map(str, DEFAULT_MESHES))}",
    )
    parser.add_argument(
        "--hc-case",
        default="default_case",
        help="region=up 且 mode=hc 时 load_parameters 的算例名（config/cases/<name>.json）",
    )
    parser.add_argument(
        "--down-case",
        default="initial_case_DOWN",
        help="region=down 时 create_standard_case_DOWN(case_type)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="汇总 CSV（utf-8-sig；列见模块文档）；默认 output/grid_independence_<region>_<mode>.csv",
    )
    parser.add_argument("--bvp-verbose", type=int, default=0, help="仅 region=up 的 BVP：FurnaceModel.solve_bvp verbose")
    parser.add_argument("--progress-log", default=None, help="过程日志；默认 logs/grid_independence_<region>_<mode>.log")
    parser.add_argument("--no-console-log", action="store_true", help="仅写进度日志文件")
    args = parser.parse_args()

    mode = _normalize_mode(args.mode)
    region = args.region

    ensure_dirs()
    meshes = parse_meshes(args.meshes)
    tag = f"{region}_{mode}"
    log_csv = Path(args.log) if args.log else output_path(f"grid_independence_{tag}.csv")
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    progress_log = (
        Path(args.progress_log) if args.progress_log else logs_path(f"grid_independence_{tag}.log")
    )
    configure_progress_logging(progress_log, console=not args.no_console_log)

    tmp_dir = ROOT / "tmp" / f"grid_independence_{tag}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    criteria = Criteria(
        max_rms_le=1e-3,
        bc_l2_le=1e-6,
        rel_le=1e-3,
        abs_le_fractions=3e-4,
    )

    t_run0 = time.perf_counter()
    LOG.info("=" * 72)
    LOG.info(
        "start | argv=%s",
        " ".join(sys.argv),
    )
    LOG.info(
        "config | region=%s mode=%s meshes=%s mesh_count=%d",
        region,
        mode,
        meshes,
        len(meshes),
    )
    LOG.info(
        "paths | csv_out=%s | progress_log=%s | tmp_dir=%s",
        log_csv.resolve(),
        progress_log.resolve(),
        tmp_dir.resolve(),
    )
    if region == "up" and mode == "hc":
        LOG.info("hc_case=%s", args.hc_case)
    if region == "down":
        LOG.info("down_case=%s", args.down_case)
    if region == "up" and mode == "bvp":
        LOG.info("bvp_verbose=%s", args.bvp_verbose)
    LOG.info(
        "criteria | BVP require_success=%s max_rms<=%s bc_l2<=%s | "
        "stability rel_outlet<=%s abs_fractions<=%s",
        criteria.require_bvp_success,
        criteria.max_rms_le,
        criteria.bc_l2_le,
        criteria.rel_le,
        criteria.abs_le_fractions,
    )
    if mode == "hc" and HC_REL_TOL_MAIN is not None:
        LOG.info(
            "criteria | HC test_hc_* outer: per-variable rel err must be < HC_REL_TOL_MAIN=%s",
            HC_REL_TOL_MAIN,
        )
    LOG.info("=" * 72)

    rows: list[dict] = []
    keys = outlet_keys(region)

    def _log_fail_detail(tail: dict) -> None:
        st = str(tail.get("status", ""))
        if st == "success":
            return
        snippet = st if len(st) <= 400 else st[:400] + "…"
        LOG.warning("run failed | mesh=%s detail=%s", tail.get("initial_mesh"), snippet)

    if region == "up" and mode == "bvp":
        base = FurnaceParameters()

        for idx, m in enumerate(meshes, start=1):
            LOG.info(
                "mesh %d/%d | begin up BVP | initial_mesh=%d | chdir_tmp=%s",
                idx,
                len(meshes),
                m,
                tmp_dir.resolve(),
            )
            rows.append(run_one_bvp_up(m, base, tmp_dir, bvp_verbose=args.bvp_verbose))
            tail = rows[-1]
            eq = is_equation_converged(tail, criteria)
            _log_fail_detail(tail)
            LOG.info(
                "mesh %d/%d | end up BVP | status=%s | equation_converged=%s | elapsed=%.2fs | "
                "bvp_success=%s bvp_tol=%s bvp_n_nodes=%s | T_out=%s t_out=%s",
                idx,
                len(meshes),
                tail["status"],
                eq,
                tail["elapsed_s"],
                tail.get("bvp_success"),
                tail.get("bvp_tol_final"),
                tail.get("bvp_n_nodes_final"),
                _fmt_outlet_log(tail.get("T_out")),
                _fmt_outlet_log(tail.get("t_out")),
            )
            print(f"mesh={m:4d}  status={tail['status']}  elapsed={tail['elapsed_s']:.1f}s")
            write_grid_independence_progress_csv(log_csv, rows)

    elif region == "down" and mode == "bvp":
        base = create_standard_case_DOWN(args.down_case)
        for idx, m in enumerate(meshes, start=1):
            LOG.info(
                "mesh %d/%d | begin down BVP | initial_mesh=%d | chdir_tmp=%s",
                idx,
                len(meshes),
                m,
                tmp_dir.resolve(),
            )
            rows.append(run_one_bvp_down(m, base, tmp_dir))
            tail = rows[-1]
            eq = is_equation_converged(tail, criteria)
            _log_fail_detail(tail)
            LOG.info(
                "mesh %d/%d | end down BVP | status=%s | equation_converged=%s | elapsed=%.2fs | "
                "bvp_success=%s bvp_tol=%s bvp_n_nodes=%s | T_out=%s p_bottom=%s",
                idx,
                len(meshes),
                tail["status"],
                eq,
                tail["elapsed_s"],
                tail.get("bvp_success"),
                tail.get("bvp_tol_final"),
                tail.get("bvp_n_nodes_final"),
                _fmt_outlet_log(tail.get("T_out")),
                _fmt_outlet_log(tail.get("p_bottom")),
            )
            print(f"mesh={m:4d}  status={tail['status']}  elapsed={tail['elapsed_s']:.1f}s")
            write_grid_independence_progress_csv(log_csv, rows)

    elif region == "up" and mode == "hc":
        for idx, m in enumerate(meshes, start=1):
            LOG.info(
                "mesh %d/%d | begin up HC | initial_mesh=%d | chdir_tmp=%s",
                idx,
                len(meshes),
                m,
                tmp_dir.resolve(),
            )
            rows.append(run_one_hc_up(m, args.hc_case, tmp_dir))
            tail = rows[-1]
            eq = is_equation_converged(tail, criteria)
            _log_fail_detail(tail)
            LOG.info(
                "mesh %d/%d | end up HC | status=%s | equation_converged=%s | elapsed=%.2fs | "
                "hc_outer_converged=%s hc_max_re_final=%s | T_out=%s t_out=%s p_bottom=%s",
                idx,
                len(meshes),
                tail["status"],
                eq,
                tail["elapsed_s"],
                tail.get("hc_outer_converged"),
                tail.get("hc_max_re_final"),
                _fmt_outlet_log(tail.get("T_out")),
                _fmt_outlet_log(tail.get("t_out")),
                _fmt_outlet_log(tail.get("p_bottom")),
            )
            print(
                f"mesh={m:4d}  status={tail['status']}  hc_outer={tail.get('hc_outer_converged')}  "
                f"hc_max_re={tail.get('hc_max_re_final')}  elapsed={tail['elapsed_s']:.1f}s"
            )
            write_grid_independence_progress_csv(log_csv, rows)

    elif region == "down" and mode == "hc":
        for idx, m in enumerate(meshes, start=1):
            LOG.info(
                "mesh %d/%d | begin down HC | initial_mesh=%d | chdir_tmp=%s",
                idx,
                len(meshes),
                m,
                tmp_dir.resolve(),
            )
            rows.append(run_one_hc_down(m, args.down_case, tmp_dir))
            tail = rows[-1]
            eq = is_equation_converged(tail, criteria)
            _log_fail_detail(tail)
            LOG.info(
                "mesh %d/%d | end down HC | status=%s | equation_converged=%s | elapsed=%.2fs | "
                "hc_outer_converged=%s hc_max_re_final=%s | T_out=%s p_bottom=%s",
                idx,
                len(meshes),
                tail["status"],
                eq,
                tail["elapsed_s"],
                tail.get("hc_outer_converged"),
                tail.get("hc_max_re_final"),
                _fmt_outlet_log(tail.get("T_out")),
                _fmt_outlet_log(tail.get("p_bottom")),
            )
            print(
                f"mesh={m:4d}  status={tail['status']}  hc_outer={tail.get('hc_outer_converged')}  "
                f"hc_max_re={tail.get('hc_max_re_final')}  elapsed={tail['elapsed_s']:.1f}s"
            )
            write_grid_independence_progress_csv(log_csv, rows)

    ref_row = None
    for r in rows:
        if is_equation_converged(r, criteria):
            ref_row = r
            break

    if ref_row is None:
        msg = "没有找到满足方程收敛判据的参考网格，请放宽判据、检查算例或调整 meshes 顺序（先大后小）。"
        LOG.warning("%s | csv=%s | wall_s=%.1f", msg, log_csv, time.perf_counter() - t_run0)
        write_grid_independence_progress_csv(log_csv, rows)
        print(msg)
        print(f"部分结果已写入 {log_csv}")
        return

    LOG.info(
        "reference | initial_mesh=%s solver=%s status=%s",
        ref_row.get("initial_mesh"),
        ref_row.get("solver"),
        ref_row.get("status"),
    )

    enriched = []
    for r in rows:
        rr = dict(r)
        rr["equation_converged"] = is_equation_converged(rr, criteria)
        rr.update(outlet_error_metrics(rr, ref_row, keys))
        rr["solution_stable_vs_ref"] = is_solution_stable(rr, criteria, region)
        enriched.append(rr)
        LOG.info(
            "vs_ref | mesh=%s eq_conv=%s stable=%s rel_diff_max=%.4e abs_diff_outlet_max=%.4e",
            rr.get("initial_mesh"),
            rr["equation_converged"],
            rr["solution_stable_vs_ref"],
            rr.get("rel_diff_outlet_max"),
            rr.get("abs_diff_outlet_max"),
        )

    df = pd.DataFrame(enriched)
    write_grid_independence_summary_csv(log_csv, enriched)

    ok = df[(df["equation_converged"] == True) & (df["solution_stable_vs_ref"] == True)]
    if ok.empty:
        msg = "没有找到同时满足“方程收敛 + 出口稳定”的网格，请放宽稳定性阈值或提高参考 mesh。"
        LOG.warning("%s | csv=%s | wall_s=%.1f", msg, log_csv, time.perf_counter() - t_run0)
        print(msg)
        print(f"结果已写入 {log_csv}")
        return

    recommended = int(ok.sort_values("initial_mesh").iloc[0]["initial_mesh"])
    ref_mesh = int(ref_row["initial_mesh"])

    LOG.info(
        "done | region=%s mode=%s reference_mesh=%s recommended_min_mesh=%s | csv=%s | wall_s=%.1f",
        region,
        mode,
        ref_mesh,
        recommended,
        log_csv,
        time.perf_counter() - t_run0,
    )
    print()
    print("=== 网格无关性结论 ===")
    print(f"region = {region}   mode = {mode}")
    print(f"reference mesh = {ref_mesh} (满足方程收敛判据)")
    print(f"recommended minimal initial_mesh = {recommended}")
    print(f"详细结果：{log_csv}")
    print(f"过程日志：{progress_log}")


if __name__ == "__main__":
    main()
