"""
将 CSV 中指定变量相对横轴列绘制成曲线（多子图）。

默认横轴为 z（与 data/ref_bvp.csv 等剖面一致）；未指定 --cols 时，自动选取除横轴外
所有数值列。

支持将上半、下半两段剖面 CSV 按横轴拼接后作图（与 test_convergence_ref_profiles 中
BVP 参考剖面拼接规则一致：可选 --z-patch 时对下半补 y/w，并对 z>=z_patch 强制 y=1-x、w=0）。

用法示例：
  python scripts/plot_csv_profiles.py data/ref_bvp.csv
  python scripts/plot_csv_profiles.py data/ref_hc.csv --cols T,t,fs,p
  python scripts/plot_csv_profiles.py path.csv --x z --output output/my_plot.png
  python scripts/plot_csv_profiles.py data/ref_bvp.csv --show
  python scripts/plot_csv_profiles.py data/initial_case_bvp_0.0-4.2m_loop.csv \\
      --csv-down data/initial_case_DOWN_bvp_4.2-5.9m_loop.csv --z-patch 4.2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _matplotlib_backend() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--show", action="store_true")
    ns, _ = p.parse_known_args()
    return "TkAgg" if ns.show else "Agg"


import matplotlib

matplotlib.use(_matplotlib_backend())
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (SRC, ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from paths import ensure_dirs, output_path


def _numeric_columns_except_x(df: pd.DataFrame, x: str) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c == x:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() > 0:
                df[c] = coerced
                cols.append(c)
    return cols


def merge_profile_halves(
    df_up: pd.DataFrame,
    df_down: pd.DataFrame,
    *,
    x_col: str,
    z_patch: float | None,
) -> pd.DataFrame:
    """
    纵向拼接上下两半剖面：按 x_col 排序，同 x 保留先出现的行（通常上半在前）。

    若给定 z_patch：下半缺 y 时用 1-x、缺 w 时用 0；拼接后对全体 z>=z_patch 行强制 y=1-x、w=0。
    """
    if x_col not in df_up.columns:
        raise ValueError(f"上半 CSV 缺少横轴列 {x_col!r}")
    if x_col not in df_down.columns:
        raise ValueError(f"下半 CSV 缺少横轴列 {x_col!r}")

    u = df_up.copy()
    d = df_down.copy()

    if z_patch is not None and "x" in d.columns:
        xv = pd.to_numeric(d["x"], errors="coerce")
        if "y" not in d.columns:
            d["y"] = 1.0 - xv
        else:
            d["y"] = pd.to_numeric(d["y"], errors="coerce").fillna(1.0 - xv)
        if "w" not in d.columns:
            d["w"] = 0.0
        else:
            d["w"] = pd.to_numeric(d["w"], errors="coerce").fillna(0.0)

    # 列并集：先保持上半列序，再追加下半独有列
    col_order: list[str] = list(u.columns)
    for c in d.columns:
        if c not in col_order:
            col_order.append(c)
    for c in col_order:
        if c not in u.columns:
            u[c] = np.nan
        if c not in d.columns:
            d[c] = np.nan
    u = u[col_order]
    d = d[col_order]

    full = pd.concat([u, d], ignore_index=True)
    full[x_col] = pd.to_numeric(full[x_col], errors="coerce")
    full = full.sort_values(x_col).reset_index(drop=True)
    full = full.drop_duplicates(subset=[x_col], keep="first")

    if z_patch is not None and "x" in full.columns:
        m = full[x_col].to_numpy(dtype=float) >= float(z_patch)
        if np.any(m) and "y" in full.columns:
            xv = full.loc[m, "x"].to_numpy(dtype=float)
            full.loc[m, "y"] = 1.0 - xv
        if np.any(m) and "w" in full.columns:
            full.loc[m, "w"] = 0.0

    return full


def plot_df(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_cols: list[str],
    out_path: Path,
    subplot_cols: int,
    figsize_per: tuple[float, float],
    dpi: int,
    title: str | None,
    show: bool,
) -> None:
    if x_col not in df.columns:
        raise ValueError(f"横轴列不存在: {x_col!r}，当前列为: {list(df.columns)}")

    for c in y_cols:
        if c not in df.columns:
            raise ValueError(f"变量列不存在: {c!r}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df = df.dropna(subset=[x_col] + y_cols, how="any")
    if df.empty:
        raise ValueError("横轴或变量列全为缺失值，无法绘图")

    df = df.sort_values(x_col)
    x = df[x_col].to_numpy(dtype=float)

    n = len(y_cols)
    ncols = max(1, int(subplot_cols))
    nrows = max(1, math.ceil(n / ncols))
    fw, fh = figsize_per
    fig_w = fw * ncols
    fig_h = fh * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    if title:
        fig.suptitle(title)

    for i, name in enumerate(y_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        y = df[name].to_numpy(dtype=float)
        ax.plot(x, y, "-", linewidth=1.2)
        ax.set_xlabel(x_col)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 变量沿横轴绘曲线（多子图）")
    parser.add_argument("csv", type=Path, help="输入 CSV 路径（上半/整段；与 --csv-down 联用时为上半）")
    parser.add_argument(
        "--csv-down",
        type=Path,
        default=None,
        metavar="PATH",
        help="下半剖面 CSV；给定则与第一个参数拼接后再绘图",
    )
    parser.add_argument(
        "--z-patch",
        type=float,
        default=None,
        metavar="Z",
        help="拼接时启用 BVP 惯例：下半补 y=1-x、w=0；全表 z>=Z 处强制 y=1-x、w=0（米）",
    )
    parser.add_argument(
        "--x",
        type=str,
        default="z",
        metavar="COL",
        help="横轴列名（默认 z）",
    )
    parser.add_argument(
        "--cols",
        type=str,
        default="",
        help="要绘制的列，逗号分隔；省略则使用除横轴外所有数值列",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出图片路径；默认 output/<csv stem>_profiles.png",
    )
    parser.add_argument(
        "--subplot-cols",
        type=int,
        default=3,
        help="子图网格每行列数（默认 3）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="输出 PNG 分辨率（默认 150）",
    )
    parser.add_argument(
        "--figsize-cell",
        type=float,
        nargs=2,
        default=[4.0, 3.0],
        metavar=("W", "H"),
        help="单个子图的宽高（英寸），整图按网格放大（默认 4 3）",
    )
    parser.add_argument("--title", type=str, default="", help="图总标题（可选）")
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存后弹窗显示（需图形界面；无界面时请省略此项）",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    ensure_dirs()
    df = pd.read_csv(csv_path)
    df_down_path = args.csv_down
    if df_down_path is not None:
        if not df_down_path.is_file():
            raise FileNotFoundError(f"下半 CSV 不存在: {df_down_path}")
        df = merge_profile_halves(
            df,
            pd.read_csv(df_down_path),
            x_col=args.x,
            z_patch=args.z_patch,
        )

    if args.cols.strip():
        y_cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    else:
        y_cols = _numeric_columns_except_x(df, args.x)
    if not y_cols:
        raise ValueError("没有可绘制的变量列（请检查 --x 与 --cols）")

    if args.output is None:
        if df_down_path is not None:
            out_path = output_path(f"{csv_path.stem}_{df_down_path.stem}_profiles_merged.png")
        else:
            out_path = output_path(f"{csv_path.stem}_profiles.png")
    else:
        out_path = args.output

    w, h = args.figsize_cell
    plot_df(
        df,
        x_col=args.x,
        y_cols=y_cols,
        out_path=out_path,
        subplot_cols=args.subplot_cols,
        figsize_per=(w, h),
        dpi=args.dpi,
        title=args.title or None,
        show=args.show,
    )
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
