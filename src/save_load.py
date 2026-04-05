# save_load.py
import json
import os
from pathlib import Path

from parameters import FurnaceParameters
from parameters_DOWN import FurnaceParameters as FurnaceParametersDOWN
from paths import cases_dir, cases_path


def _case_stem(filename: str) -> str:
    s = str(filename).strip()
    if s.endswith(".json"):
        s = s[: -len(".json")]
    return s


def resolve_case_json_path(case_name: str) -> Path:
    """``config/cases/<stem>.json`` 的绝对路径。"""
    stem = _case_stem(case_name)
    return cases_path(f"{stem}.json")


def save_parameters(params, filename=None):
    """保存参数到 ``config/cases/<name>.json``。"""
    if filename is None:
        filename = params.case_name

    stem = _case_stem(filename)
    cases_dir().mkdir(parents=True, exist_ok=True)
    filepath = cases_path(f"{stem}.json")

    data = {}
    for key, value in params.__dict__.items():
        data[key] = value

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"参数已保存: {filepath}")
    return str(filepath)


def load_parameters(filename):
    """从 ``config/cases`` 加载参数。"""
    filepath = resolve_case_json_path(filename)

    if not filepath.is_file():
        raise FileNotFoundError(f"参数文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = FurnaceParameters(data["case_name"])
    for key, value in data.items():
        if hasattr(params, key):
            setattr(params, key, value)

    print(f"参数已加载: {filepath}")
    return params


def save_parameters_down(params, filename=None):
    """保存下半 ``parameters_DOWN.FurnaceParameters`` 到 ``config/cases/<name>.json``。"""
    if filename is None:
        filename = params.case_name

    stem = _case_stem(filename)
    cases_dir().mkdir(parents=True, exist_ok=True)
    filepath = cases_path(f"{stem}.json")

    data = dict(params.__dict__)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"参数已保存: {filepath}")
    return str(filepath)


def load_parameters_down(filename):
    """从 ``config/cases`` 加载下半参数（JSON）。"""
    filepath = resolve_case_json_path(filename)

    if not filepath.is_file():
        raise FileNotFoundError(f"参数文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = FurnaceParametersDOWN(data["case_name"])
    for key, value in data.items():
        if hasattr(params, key):
            setattr(params, key, value)

    print(f"参数已加载: {filepath}")
    return params


def list_saved_cases():
    """列出 ``config/cases`` 中的算例名（.json 去掉后缀）。"""
    d = cases_dir()
    if not d.is_dir():
        return []
    names: list[str] = []
    for file in os.listdir(d):
        if file.endswith(".json"):
            names.append(file[:-5])
    return sorted(names)
