import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import numpy as np
import pickle, matplotlib.pyplot as plt, pandas as pd
import xarray as xr

def _central_lon(lon0, lon1):
    cen = (lon0 + lon1) / 2.0
    return (cen + 180) % 360 - 180          # 映射到 (-180, 180]

def _wrap(lon, center):
    """把经度映射到 (center-180, center+180]"""
    return (lon - center + 180) % 360 - 180

def lon_trans(lon):
    """把经度映射到 (center-180, center+180]"""
    return (lon + 180) % 360 - 180

def draw_map_centered(
    da: xr.DataArray,
    *,
    extent,            # (lon0, lon1, lat0, lat1) —— 0–360 或 (-180, 180] 都可
    ax=None,
    cmap,
    levels,
    vmin, vmax,
    title=None,title_size=16,
    map_xticks=[], map_yticks=[],
    tick_params_size=None,
    axis_off=True,
    methods="contourf"
):
    lon0, lon1, lat0, lat1 = extent
    cen = _central_lon(lon0, lon1)

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=cen))

    # ---- 关键：把 lon0/lon1 转到“±180 around cen” ----
    # lon0_w = _wrap(lon0, cen)
    # lon1_w = _wrap(lon1, cen)

    if len(map_xticks):
        ax.set_xticks(lon_trans(map_xticks), crs=ccrs.PlateCarree())
    if len(map_yticks):
        ax.set_yticks(lon_trans(map_yticks), crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter(number_format='.0f', dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.0f', )
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    if tick_params_size:
        ax.tick_params(axis='both', labelsize=tick_params_size)

    # ax.set_extent([lon0_w, lon1_w, lat0, lat1], crs=ccrs.PlateCarree())
    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
    if methods == "contourf":
        im = ax.contourf(
            da["lon"], da["lat"], da,
            transform=ccrs.PlateCarree(),
            cmap=cmap, levels=levels,
            vmin=vmin, vmax=vmax, extend="both", corner_mask=False
        )
    elif methods == "pcolormesh":
        im = ax.pcolormesh(
            da["lon"], da["lat"], da,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin, vmax=vmax,  shading="auto", zorder=0
        )
    else:
        raise ValueError("Unexpected methods")


    ax.add_feature(cfeature.LAND, facecolor="beige", edgecolor="k", linewidth=0.6)

    # ax.axis("off")
    if axis_off:
        ax.spines['geo'].set_visible(False)
    if title: ax.set_title(title, fontsize = title_size)
    return im


def plot_var_map(
    da,                    # xarray.DataArray (lat, lon)  方差或任何空间场
    ax: plt.Axes | None = None,
    *,
    cmap="cmo.balance",
    levels: int | list = 15,
    land_color="beige",
    tick_step=60,
    extent=None,          # (lon_min, lon_max, lat_min, lat_max)
    add_colorbar=True,
    cbar_kwargs=None,
    title=None,
    coast_kwargs=None,
):
    """
    画空间方差/距平图 —— 单函数即可重复用在多季节、多组别循环

    Parameters
    ----------
    da           : 已经选择好 lat/lon 的 DataArray
    ax           : 传入已有子图；None → 自动建新图
    cmap, levels : 等值面属性
    tick_step    : 经纬度刻度间隔 (°)
    extent       : 地图裁剪范围 (lon0, lon1, lat0, lat1)
    """

    if ax is None:
        fig = plt.figure(figsize=(7,4))
        ax  = plt.axes(projection=ccrs.PlateCarree())

    # ----- 栅格数据 -----
    lon = da["lon"]
    lat = da["lat"]
    data = da.values

    # ----- 画底图 -----
    pcm = ax.contourf(
        lon, lat, data,
        levels=levels,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=0,
    )

    # ----- 海陆 & 海岸线 -----
    ax.add_feature(
        cfeature.LAND,
        facecolor=land_color,
        edgecolor="k",
        linewidth=0.7,
        zorder=1,
        **(coast_kwargs or {})
    )

    # ----- 轴域范围 & 刻度 -----
    if extent:
        ax.set_extent(extent, ccrs.PlateCarree())

    # 刻度 & 格式化
    lon_ticks = np.arange(-180, 181, tick_step)
    lat_ticks = np.arange(-90,  91, tick_step)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", dateline_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))
    ax.tick_params(labelsize=12)

    # ----- 标题 & colorbar -----
    if title:
        ax.set_title(title, fontsize=14)

    if add_colorbar:
        cbar_kwargs = cbar_kwargs or {}
        plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, **cbar_kwargs)

    return ax

def plot_time_series(
    time, values,
    *,
    ax=None,
    label=None,
    show="year",          # "year" | "year-month"
    ylabel=None,
    title=None,
    color=None,
    grid=True,
):
    """
    Parameters
    ----------
    time    : 1-D datetime64 / pandas.DatetimeIndex / list
    values  : 1-D 数值
    show    : x 轴标签格式
              • "year"        → 2000, 2005, ...
              • "year-month"  → 2000-01, 2000-07, ...
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(time, values, color=color, label=label, lw=1.5)

    # ---------- x 轴格式 ----------
    locator = AutoDateLocator()
    if show == "year":
        formatter = DateFormatter("%Y")
    else:  # "year-month"
        formatter = DateFormatter("%Y-%m")

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=45)

    if ylabel: ax.set_ylabel(ylabel)
    if title : ax.set_title(title)
    if grid  : ax.grid(ls="--", alpha=0.4)
    if label : ax.legend(frameon=False)

    plt.tight_layout()
    return ax

def plot_seasonality(
    months, values,
    *,
    ax=None,
    label=None,
    ylabel=None,
    title=None,
    style="-o",
    month_fmt="short",    # "short"=JFMAMJ... / "full"=Jan/Feb/Mar...
    grid=True,
):
    """
    Parameters
    ----------
    months : 1-D 1..12
    values : 1-D 数值
    month_fmt : "short" → J F ...  /  "full" → Jan Feb ...
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))

    ax.plot(months, values, style, label=label, lw=1.5)

    month_labels_short = list("JFMAMJJASOND")
    month_labels_full  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    labels = month_labels_short if month_fmt == "short" else month_labels_full
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(labels)

    if ylabel: ax.set_ylabel(ylabel)
    if title : ax.set_title(title)
    if grid  : ax.grid(ls="--", alpha=0.4)
    if label : ax.legend(frameon=False)

    plt.tight_layout()
    return ax

# ----------------------------------------------------------
# 3. 功率谱 / 频谱图
# ----------------------------------------------------------
def plot_power_spectrum(
    freq, psd,
    *,
    ax=None,
    label=None,
    logx=False,
    logy=True,
    xlabel="Frequency",
    ylabel="Power",
    title=None,
    color=None,
    grid=True,
):
    """
    Parameters
    ----------
    freq : 频率轴 (e.g., cycles / yr, Hz)
    psd  : 功率谱密度
    logx / logy : 是否把 x / y 轴设为对数
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))

    ax.plot(freq, psd, color=color, label=label, lw=1.5)

    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if grid : ax.grid(ls="--", alpha=0.4, which="both")
    if label: ax.legend(frameon=False)

    plt.tight_layout()
    return ax