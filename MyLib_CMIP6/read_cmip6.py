from __future__ import annotations
import os
from dataclasses import dataclass
import xarray as xr
from pathlib import Path
from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
import xeofs as xe
import pickle

def extract_model_name(path: Path | str) -> str:
    """
    ACCESS-CM2_r1i1p1f1.nc → ACCESS-CM2
    """
    return Path(path).stem.split("_")[0]

@dataclass
class ExperimentFile:
    model_name: str       # 模型名，对应 G:\tos 文件夹下的子文件夹名
    experiment_name: str  # 实验名（这里固定为 r1i1p1f1）
    date: str             # 日期，提取自 gn 文件夹下 vyyyymmdd 子文件夹（例如 '20191108'）
    file_path: str        # 数据文件的完整路径

class FileStore:
    def __init__(self):
        self.files: List[ExperimentFile] = []  # 存放 ExperimentFile 对象的列表

    def add_file(self, exp_file: ExperimentFile):
        self.files.append(exp_file)

    def get_by_model(self, model_name: str) -> List[ExperimentFile]:
        """根据模型名返回对应的所有文件"""
        return [f for f in self.files if f.model_name == model_name]

    def __repr__(self):
        return "\n".join(str(f) for f in self.files)


# ------------------------ CMIP6主类 ------------------------ #

    # 要求的包括: PO and IO SSTv (lon*lat的矩阵)
    # PO和IO地区 的 EOF1 和 2
    # ENSO and IOD time series (1 * time 的序列)
    # seasonality
    # power spec
PathLike = Union[str, Path]
class CMIPGroupStats:
    """
    统计多组 CMIP 文件（每组含多个 member）：
    • member → 先算时序方差 / ENSO / IOD / EOF
    • group → 对 member 取平均
    ☆ 结果缓存：
        variance_member[group] : (member, lat, lon)
        variance_mean[group]   : (lat, lon)      … group-mean
        enso_member[group]     : (time, member)
        enso_mean[group]       : (time)
        iod_member / iod_mean  : 同上
        eof_maps_member[group] : (member, mode, lat, lon)
        eof_maps_mean[group]   : (mode, lat, lon)
        pcs_member[group]      : (time, member, mode)
        pcs_mean[group]        : (time, mode)
    """
    # ---------- 初始化 ---------- #
    def __init__(
        self,
        groups: Dict[str, List[PathLike]],
        var_name: str = "tos",
        chunks: Dict[str, int] | None = None,
        member_id_func=None,                     # 自定义文件 → member 名函数
    ):
        self.groups = {g: [Path(p) for p in v] for g, v in groups.items()}
        self.var   = var_name
        self.chunks = chunks or {}
        self.member_id_func = (
            member_id_func or (lambda p: p.stem)  # 默认用文件名（无后缀）
        )

        # 结果缓存
        self.variance_member: Dict[str, xr.DataArray] = {}
        self.variance_mean  : Dict[str, xr.DataArray] = {}
        #enso iod index缓存
        self.enso_member    : Dict[str, xr.DataArray] = {}
        self.enso_mean      : Dict[str, xr.DataArray] = {}
        self.iod_member     : Dict[str, xr.DataArray] = {}
        self.iod_mean       : Dict[str, xr.DataArray] = {}
        #eof缓存
        self.eof_maps_member: Dict[str, Dict[str, xr.DataArray]] = {}
        self.eof_maps_mean  : Dict[str, Dict[str, xr.DataArray]] = {}
        self.pcs_member     : Dict[str, Dict[str, xr.DataArray]] = {}
        self.pcs_mean       : Dict[str, Dict[str, xr.DataArray]] = {}
        self.evr_mean       : Dict[str, Dict[str, xr.Dataset]] = {}

        # ---------- 私有：读取单文件并加 member 维 ---------- #
    def _open_one(self, path: PathLike) -> xr.DataArray:
        """
        • 读取单文件 → 统一时间坐标到“月首 00:00”
        • 去掉同月重复记录（保留第一条）
        • 加 member 维度
        """
        member = self.member_id_func(path)

        # ---- ① 读取 ----
        da = xr.open_dataset(
            path,
            chunks=self.chunks or None,
            decode_times=True,
        )[self.var]

        # ---- ② 只改坐标，不做平均 ----
        # 把 1940-01-16 12:00 / 1940-01-17 00:00 → 1940-01-01 00:00
        idx_month = (
            da.indexes["time"]  # pandas.DatetimeIndex
            .to_period("M")  # 1940-01
            .to_timestamp()  # 1940-01-01 00:00
        )
        da = da.assign_coords(time=("time", idx_month))

        # 去重：同月只保留第一条记录（速度远快于 resample.mean）
        _, first = np.unique(da["time"].values, return_index=True)
        da = da.isel(time=first)

        # ---- ③ 加 member 维 ----
        da = da.expand_dims(member=[member])

        return da

    # ---------- 1. 方差 ---------- #
    def calc_variance(
        self,
        group: str,
        sel_kwargs: Optional[Dict] = None,
        area_weighted: bool = False,
    ):
        if group in self.variance_member:
            return self.variance_mean[group]

        var_list = []
        for p in self.groups[group]:
            da = self._open_one(p)
            if sel_kwargs:
                da = da.sel(**sel_kwargs)

            if area_weighted and "lat" in da.dims:
                w = np.cos(np.deg2rad(da["lat"]))
                da = da.weighted(w)

            var = da.var("time", skipna=True, keep_attrs=False).compute()
            var_list.append(var)
            da.close()

        var_mem = xr.concat(var_list, dim="member")            # (member, lat, lon)
        var_mean = var_mem.mean("member")

        self.variance_member[group] = var_mem
        self.variance_mean[group]   = var_mean
        return var_mean

    # ---------- 2. ENSO & IOD ---------- #
    def calc_enso_iod(self, group: str):
        if group in self.enso_member:
            return self.enso_mean[group], self.iod_mean[group]

        nino_list, iod_list = [], []
        for p in self.groups[group]:
            da = self._open_one(p)
            w  = np.cos(np.deg2rad(da["lat"]))

            # Niño3.4 (5N–5S, 170W–120W = 190–240E)
            n34 = (
                da.sel(lat=slice(-5, 5), lon=slice(190, 240))
                  .weighted(w)
                  .mean(("lat", "lon"), skipna=True)
                  .compute()
            )

            # IOD = WIO − SEIO
            wio = (
                da.sel(lat=slice(-10, 10), lon=slice(50, 70))
                  .weighted(w)
                  .mean(("lat", "lon"), skipna=True)
                  .compute()
            )
            seio = (
                da.sel(lat=slice(-10, 0), lon=slice(90, 110))
                  .weighted(w)
                  .mean(("lat", "lon"), skipna=True)
                  .compute()
            )
            iod = (wio - seio)

            nino_list.append(n34)
            iod_list.append(iod)
            da.close()

        enso_mem = xr.concat(nino_list, dim="member")          # (time, member)
        iod_mem  = xr.concat(iod_list,  dim="member")
        enso_mean = enso_mem.mean("member")
        iod_mean  = iod_mem.mean("member")

        self.enso_member[group] = enso_mem
        self.enso_mean[group]   = enso_mean
        self.iod_member[group]  = iod_mem
        self.iod_mean[group]    = iod_mean
        return enso_mean, iod_mean

    # ---------- 3. EOF & PC ---------- #
    def calc_eofs(
            self,
            group: str,
            region: str,
            sel_kwargs: dict,
            n_modes: int = 2,
            area_weighted: bool = True,
    ):
        """
        先对 group 内每个 member 计算 EOF，再在 member 维做平均。
        结果按 [group][region] 双层缓存。
        """
        # ---------- 初始化该 group 的 region 字典 ---------- #
        for store in (
                self.eof_maps_member,
                self.eof_maps_mean,
                self.pcs_member,
                self.pcs_mean,
        ):
            store.setdefault(group, {})

        # ---------- 已算过直接返回 ---------- #
        if region in self.eof_maps_member[group]:
            return (
                self.eof_maps_mean[group][region],
                self.pcs_mean[group][region],
            )

        # ---------- 对每个 member 逐一计算 ---------- #
        eofs_list, pcs_list = [], []

        for path in self.groups[group]:
            da = self._open_one(path).sel(**sel_kwargs)  # 加 member 维

            # (可选) √cosφ 权重
            weights = (
                np.sqrt(np.cos(np.deg2rad(da["lat"]))) if area_weighted else None
            )

            solver = xe.single.EOF(n_modes=n_modes)
            solver.fit(da, dim="time", weights=weights)

            eofs = solver.components().compute()
            pcs = solver.scores().compute()

            eofs_list.append(eofs)
            pcs_list.append(pcs)
            da.close()

        # ---------- 拼 member 维 & 求平均 ---------- #
        eofs_mem = xr.concat(eofs_list, dim="member")  # (member, mode, lat, lon)
        pcs_mem = xr.concat(pcs_list, dim="member")  # (time, member, mode)

        # 若需同号对齐，可在此处做符号一致化；此处简单平均
        eofs_mean = eofs_mem.mean("member")
        pcs_mean = pcs_mem.mean("member")

        # ---------- 缓存 ---------- #
        self.eof_maps_member[group][region] = eofs_mem
        self.eof_maps_mean[group][region] = eofs_mean
        self.pcs_member[group][region] = pcs_mem
        self.pcs_mean[group][region] = pcs_mean

        return eofs_mean, pcs_mean

    # ---------- 批量接口 ---------- #
    def batch_variance(self, **sel_kwargs):
        for g in self.groups:
            self.calc_variance(g, sel_kwargs)
        return self.variance_mean

    def batch_enso_iod(self):
        for g in self.groups:
            self.calc_enso_iod(g)
        return self.enso_mean, self.iod_mean

    def batch_eofs_multi(self, regions: Dict[str, Dict], n_modes=2):
        """
        regions = {
            "PO": dict(lat=slice(-30,30), lon=slice(120,290)),
            "IO": dict(lat=slice(-30,30), lon=slice(40,115)),
        }
        """
        for g in self.groups:
            for r, kw in regions.items():
                self.calc_eofs(g, region=r, sel_kwargs=kw, n_modes=n_modes)

    def explained_variance_from_pcs(pcs: xr.DataArray) -> xr.DataArray:
        """
        pcs : (time, mode) 或 (time, member, mode)
        返回 λ_k 以及 evr_k
        """
        # 若还有 member 维，先在 time 维求 var，再对 member 平均
        if "member" in pcs.dims:
            lam = pcs.var("time", ddof=1).mean("member")  # (mode)
        else:
            lam = pcs.var("time", ddof=1)  # (mode)

        evr = lam / lam.sum("mode")
        lam.name = "eigenvalue"
        evr.name = "explained_variance_ratio"
        return xr.Dataset({"eigenvalue": lam, "evr": evr})

    def calc_evr(
            self,
            group: str,
            region: str,
            *,
            sel_kwargs: dict | None = None,
            overwrite: bool = False,
    ):
        """
        计算指定 group-region 的 eigenvalue & EVR。
        如果已存在且 overwrite=False，直接返回缓存。
        若 PCs 不存在，则按 sel_kwargs 先调用 calc_eofs() 补齐。
        """
        # --- 已缓存 & 不覆盖 ---
        if (not overwrite and
                group in self.evr_mean and region in self.evr_mean[group]):
            return self.evr_mean[group][region]

        # --- 保证 PCs 已存在 ---
        if group not in self.pcs_mean or region not in self.pcs_mean[group]:
            if sel_kwargs is None:
                raise KeyError(
                    f"{group}-{region} 没有 PC；请提供 sel_kwargs 让我先算 EOF/PC"
                )
            self.calc_eofs(group, region, sel_kwargs)

        pcs = self.pcs_mean[group][region]  # (time, mode [, member])
        lam = pcs.var("time", ddof=1)
        if "member" in lam.dims:  # 多成员 → 取平均
            lam = lam.mean("member")
        evr = lam / lam.sum("mode")

        ds = xr.Dataset({"eigenvalue": lam, "evr": evr})

        # --- 安全写入，绝不覆盖 dict ---
        self.evr_mean.setdefault(group, {})[region] = ds
        return ds

    def batch_evr(
            self,
            regions: dict[str, dict],  # {"PO": sel_kwargs, "IO": sel_kwargs, ...}
            *,
            overwrite: bool = False,
    ):
        """
        给所有 group × regions 计算 EVR，自动补缺 PC / EOF。
        """
        for g in self.groups:
            for r, kw in regions.items():
                self.calc_evr(g, r, sel_kwargs=kw, overwrite=overwrite)
        return self.evr_mean

        # ---------- 打印 ---------- #
    def __repr__(self):
        return (
            f"<CMIPGroupStats groups={list(self.groups)}  var='{self.var}'>"
        )

# --------------------------------------------------
    def save_pickle(self, file_path: PathLike, overwrite: bool = False) -> None:
        """
        将整个 CMIPGroupStats 实例序列化为 .pkl
        （含已计算好的 variance/EOF/ENSO/IOD 缓存）
        """
        file_path = Path(file_path)
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"{file_path} already exists (use overwrite=True).")

        with file_path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✔ stats object saved → {file_path.resolve()}")


    @staticmethod
    def load_pickle(file_path: PathLike) -> "CMIPGroupStats":
        """
        反序列化：直接取回完整 stats 对象
        """
        file_path = Path(file_path)
        with file_path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, CMIPGroupStats):
            raise TypeError(f"Object in {file_path} is not a CMIPGroupStats instance.")
        print(f"✔ stats object loaded ← {file_path.resolve()}")
        return obj