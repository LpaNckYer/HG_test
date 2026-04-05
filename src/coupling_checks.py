# coupling_checks.py
"""风口耦合迭代中，对每次上半 / 下半 BVP 或 HC 求解的显式收敛门槛。"""
from __future__ import annotations


def require_bvp_segment_converged(results: dict, *, segment: str) -> None:
    """
    segment: \"up\" 上半 (~0–4.2 m)，\"down\" 下半 (~4.2–5.9 m)。
    不满足则抛错，使耦合流程立即失败。
    """
    label = "上半(0–4.2 m)" if segment == "up" else "下半(4.2–5.9 m)"
    bs = results.get("bvp_success")
    if bs is not True:
        raise RuntimeError(f"BVP {label} 求解未收敛: bvp_success={bs!r}")


def require_hc_segment_converged(results: dict, *, segment: str) -> None:
    """segment 同 BVP；依据 test_hc_4n4 / test_hc_6 返回的 hc_converged。"""
    label = "上半(0–4.2 m)" if segment == "up" else "下半(4.2–5.9 m)"
    hc = results.get("hc_converged")
    if hc is not True:
        raise RuntimeError(f"HC {label} 求解未收敛: hc_converged={hc!r}")
