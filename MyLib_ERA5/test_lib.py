from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib import rcParams#set font type of figures as Arial
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import math
import xarray as xr

class regions_2d:
    # 2d data for figuring
    def __init__(self, data_in_sst, data_in_olr, lonlat_sst, lonlat_olr, region_name=""):
        self.sst = data_in_sst
        self.olr = data_in_olr
        self.lon_olr = lonlat_olr[0]
        self.lat_olr = lonlat_olr[1]
        self.lon_sst = lonlat_sst[0]
        self.lat_sst = lonlat_sst[1]
        self.region_name = region_name
    def set_range(self, range=""):
        if range == "":
            self.range = [np.nanmin(self.lon_sst), np.nanmax(self.lon_sst), np.nanmin(self.lat_sst), np.nanmax(self.lat_sst)]
        else:
            self.range = range
    def set_clv(self, data_type, clv=np.array([0,0]), cax=np.array([0,0]), index_type= "reg"):
        if data_type == "sst":
            data_max = np.nanmax(np.abs(self.sst))
            c_data = int(data_max + 1)
            if index_type == "std":
                c_low = 0
            elif index_type == "reg":
                c_low = -c_data

            if np.std(clv) == 0.0:
                self.clv_sst = np.linspace(c_low,c_data,41)
            else:
                self.clv_sst = clv

            if np.std(cax) == 0.0:
                self.caxis_sst = np.linspace(c_low,c_data,5)
            else:
                self.caxis_sst = cax
        elif data_type == "u10":
            data_max = np.nanmax(np.abs(self.olr))
            c_data = int(data_max + 1)
            if index_type == "std":
                c_low = 0
            elif index_type == "reg":
                c_low = -c_data

            if np.std(clv) == 0.0:
                self.clv_olr = np.linspace(c_low, c_data, 21)
            else:
                self.clv_olr = clv

            if np.std(cax) == 0.0:
                self.caxis_olr = np.linspace(c_low, c_data, 5)
            else:
                self.caxis_olr = cax
        else:
            print("wrong data type")

    def set_caxis(self, c_range, c_which):
        #setting the range of sst or u10
        if c_which == "sst":
            self.caxis_sst = c_range
        elif c_which == "olr":
            self.caxis_olr = c_range

    def set_ticks(self,n):
        self.xticks = np.linspace(np.nanmin(self.lon_sst), np.nanmax(self.lon_sst), n)
        self.yticks = [-30, -15, 0, 15, 30]

def draw_fig(data_in, lon_list, lat_list, clevs1, caxis, title, fnames_out, range=[0,360,-60,60],
             xticks=[60,120,180,240,300],yticks=[-30, -15, 0, 15, 30]):
    lons = lon_list
    lats = lat_list
    lon_mean = np.mean([range[0], range[1]])

    f1 = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.2)

    ax = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=lon_mean))
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.0f', dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.0f', )

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_extent(range, ccrs.PlateCarree())

    ax.tick_params(axis='both', labelsize=24)
    fill = ax.contourf(lons, lats, data_in[:,:], clevs1, extend='both', transform=ccrs.PlateCarree(),
                       cmap=cmocean.cm.balance, zorder=0)
    ax.add_feature(cfeature.LAND, linewidth=2.0, facecolor='beige', edgecolor='k', zorder=1)
    plt.title(title, loc='center', fontsize=30)

    # colorbar part
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    cb = plt.colorbar(fill, orientation='vertical', cax=cax, ticks=caxis)
    cb.ax.tick_params(labelsize=24)

    plt.savefig(fnames_out, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(f1)

# def get_region_ds(ds_in, range0=[0, 0, 0, 0], index_name=""):
#     ds_now = ds_in
#     if index_name == "":
#         ds_out =

def draw_sst_u10(ax, region_2d_data, title="", cmcolor=cmocean.cm.balance, cb_input = "on"):

    lons_sst = region_2d_data.lon_sst
    lats_sst = region_2d_data.lat_sst
    lons_olr = region_2d_data.lon_olr
    lats_olr = region_2d_data.lat_olr

    ax.set_xticks(region_2d_data.xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(region_2d_data.yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.0f', dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.0f', )

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_extent(region_2d_data.range, ccrs.PlateCarree())

    ax.tick_params(axis='both', labelsize=24)
    fill = ax.contourf(lons_sst, lats_sst, region_2d_data.sst, region_2d_data.clv_sst, extend='both', transform=ccrs.PlateCarree(),
                       cmap=cmcolor, zorder=0)
    olr_contour = ax.contour(lons_olr, lats_olr, region_2d_data.olr,levels=region_2d_data.clv_olr, colors='k', extend='both', transform=ccrs.PlateCarree(),
                       zorder=0, linewidths=1, linestyles='-')
    #设置等高线字体
    ax.clabel(olr_contour, inline="true", fontsize=8, fmt='%.1f')

    ax.add_feature(cfeature.LAND, linewidth=2.0, facecolor='beige', edgecolor='k', zorder=1)
    # plt.title(title, loc='center', fontsize=30)
    if title!="":
        ax.set_title(title, fontsize=30)
    # colorbar part
    if cb_input == "on":
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        cb = plt.colorbar(fill, orientation='vertical', cax=cax, ticks=region_2d_data.caxis_sst)
        cb.ax.tick_params(labelsize=24)

def moving_mean(xx, windows):
    b = np.ones(windows) / windows
    xx_mean = np.convolve(xx, b, mode="same")

    n_conv = math.ceil(windows / 2)

    # 補正部分
    xx_mean[0] *= windows / n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= windows / (i + n_conv)
        xx_mean[-i] *= windows / (i + n_conv - (windows % 2))
    # size%2は奇数偶数での違いに対応するため

    return xx_mean
