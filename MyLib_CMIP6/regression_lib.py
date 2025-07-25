import math

import xarray as xr
import numpy as np
from .eof_lib import dt64_to_dt
from sklearn.linear_model import LinearRegression

#从给定ncfile中获取ENSO nino3.4 index（170-120W， 5s-5n）
# def get_enso_index(xr_in, lon0=190, lon1=240, lat0=-5, lat1=5):
def get_enso_index(xr_in, range0=[190, 240, -5, 5]):
    #选择数据范围在给定区间的数据
    xr_enso = xr_in.sel(lon=slice(range0[0], range0[1]),lat=slice(range0[2], range0[3]))#创建一个新xr数据,有三个维度 (lon,lat,time)
    #原始数据已经是异常值数据了，所以不用去climatology，直接求平均
    enso_data = xr_enso.data
    enso_index = np.nanmean(enso_data, axis=(1, 2))#只有time一个维度的xr数据
    return enso_index

# IOB ep: 90-110E, 10s-0
#     wp: 50-70E, 10s-10n
def get_DMI_index(xr_in):
    #选择数据范围在给定区间的数据
    range_ep = [90, 110, -10, 0]
    range_wp = [50, 70, -10, 10]
    xr_ep = xr_in.sel(lon=slice(range_ep[0], range_ep[1]),lat=slice(range_ep[2], range_ep[3]))#创建一个新xr数据,有三个维度 (lon,lat,time)
    xr_wp = xr_in.sel(lon=slice(range_wp[0], range_wp[1]),lat=slice(range_wp[2], range_wp[3]))
    #原始数据已经是异常值数据了，所以不用去climatology，直接求平均
    ep_data = xr_ep.data
    wp_data = xr_wp.data

    wp_index = np.nanmean(wp_data, axis=(1, 2))#只有time一个维度的xr数据
    ep_index = np.nanmean(ep_data, axis=(1, 2))
    DMI = wp_index - ep_index
    return DMI

def get_IOBM_index(xr_in, range0=[40, 100, -20, 20]):
    #选择数据范围在给定区间的数据
    xr_IOBM = xr_in.sel(lon=slice(range0[0], range0[1]),lat=slice(range0[2], range0[3]))#创建一个新xr数据,有三个维度 (lon,lat,time)
    #原始数据已经是异常值数据了，所以不用去climatology，直接求平均
    IOBM_data = xr_IOBM.data
    IOBM_index = np.nanmean(IOBM_data, axis=(1, 2))#只有time一个维度的xr数据
    return IOBM_index

def get_seasonal_data(ds_in, varname, tname="time"): # divide the data into each season
    #ds_in: input dataset
    data_in = ds_in[varname].data#排布一般为 time, lat, lon
    lon_in = ds_in["lon"].data
    lat_in = ds_in["lat"].data
    time_in = dt64_to_dt(ds_in[tname])

    lon_lenth = len(lon_in)
    lat_lenth = len(lat_in)
    time_lenth = len(time_in)
    year0 = time_in[0].year
    years = int(time_lenth/12)
    year1 = year0 + years
    years_out = np.arange(year0, year1, 1)
    # months_list = []
    # season_name = ["MAM", "JJA", "SON", "DJF"]#原始求的时候就是第一年2月到最后一年1月的数据,最后一年的要也行不要也行，反正会少一个月数据
    MAM = [3,4,5]; JJA = [6,7,8]; SON = [9,10,11]; DJF=[12,1,2]

    # 准备四个计数器，用来统计每个格子装了几个数据，方便最后做除法
    data0 = np.empty([years,lat_lenth,lon_lenth])
    datacount = np.empty([years,lat_lenth,lon_lenth],dtype = int)
    data_count_MAM = datacount; data_count_JJA = datacount; data_count_SON = datacount; data_count_DJF = datacount;
    # 准备装四个季节数据的篮子
    data_out_MAM = data0; data_out_JJA = data0;
    data_out_SON = data0; data_out_DJF = data0;
    #改造成一个有 lon,lat,year,season四个维度的数据
    for y in range(0,time_lenth):
        month_now = time_in[y].month
        year_now = time_in[y].year
        year_pos = time_in[y].year - year0
        #months_list.append(time_in[y].month)
        if month_now in MAM:
            data_out_MAM[year_pos,:,:] = data_out_MAM[year_pos,:,:] + data_in[y,:,:]
            data_count_MAM[year_pos,:,:] = data_out_MAM[year_pos,:,:] + 1
        elif month_now in JJA:
            data_out_JJA[year_pos, :, :] = data_out_JJA[year_pos, :, :] + data_in[y, :, :]
            data_count_JJA[year_pos, :, :] = data_out_JJA[year_pos, :, :] + 1
        elif month_now in SON:
            data_out_SON[year_pos, :, :] = data_out_SON[year_pos, :, :] + data_in[y, :, :]
            data_count_SON[year_pos, :, :] = data_out_SON[year_pos, :, :] + 1
        elif month_now in DJF:
            if month_now in [1,2]:
                year_pos = year_pos - 1
                year_now = year_now - 1
            if year_now in years_out:
                data_out_DJF[year_pos, :, :] = data_out_DJF[year_pos, :, :] + data_in[y, :, :]
                data_count_DJF[year_pos, :, :] = data_out_DJF[year_pos, :, :] + 1
    data_out_MAM = data_out_MAM / data_count_MAM; data_out_JJA = data_out_JJA / data_count_JJA
    data_out_SON = data_out_SON / data_count_SON; data_out_DJF = data_out_DJF / data_count_DJF
    MAM_out = xr.DataArray(data_out_MAM,dims=("year","lat","lon"),
                           coords={"year":years_out,"lat":lat_in,"lon":lon_in})
    JJA_out = xr.DataArray(data_out_JJA, dims=("year", "lat", "lon"),
                           coords={"year": years_out, "lat": lat_in, "lon": lon_in})
    SON_out = xr.DataArray(data_out_SON, dims=("year", "lat", "lon"),
                           coords={"year": years_out, "lat": lat_in, "lon": lon_in})
    DJF_out = xr.DataArray(data_out_DJF, dims=("year", "lat", "lon"),
                           coords={"year": years_out, "lat": lat_in, "lon": lon_in})
    ds_out = xr.Dataset(dict(MAM=MAM_out,JJA=JJA_out,SON=SON_out,DJF=DJF_out))
    ds_out.attrs["var_name"]=varname
    return ds_out
    # ds_in.coords["month"] = ("time", months_list)

def get_LR_coef(data_in, index_in):
    model = LinearRegression()
    index_in = index_in.reshape((-1,1))
    lr = model.fit(index_in, data_in)
    return lr.coef_

def get_LR_coef_season(ds_season_in, index_in):
    season_name = ["MAM", "JJA", "SON", "DJF"]
    xr_out_list = []
    lat_now = season_name["lat"]
    lon_now = season_name["lon"]
    for season in season_name:
        xr_now = ds_season_in[season]
        xr_out = xr_now

def season_reshape(data_in):
    data_size = data_in.shape
    years = int(data_size[0]/12)
    data_now = data_in[0:years*12,:,:]
    # 数据按年排列，从第一年1月到最后一年12月
    # 第一步先变成按月排列到数据
    # 然后
    # 第一个数据是0，第一行第二个数据是第一年一月的数据..？
    # 我靠我直接整个数据向后挪一位不就好了 我靠傻逼了
    data_box = np.full(np.shape(data_now),np.nan)
    data_box[1:,:,:] = data_now[:-1,:,:]
    data_arrange = data_box.reshape((years, 12, data_size[1], data_size[2]))
    # 顺序是 ["DJF"， "MAM", "JJA", "SON"]
    season_data = []
    for i in range(0,4):
        season_data_now = data_arrange[:,i*3:(i+1)*3,:,:]
        # season_data_now = season_data_now.reshape((3*years, data_size[1], data_size[2]))
        season_data_mean = np.nanmean(season_data_now, axis=1)
        season_data.append(season_data_mean)
    return season_data #output: a list consist of 4 ndarray

def get_season_LR_coef(season_data, index_data, lag=[0,0,0,0]):
    season_coef = []
    #lag: lag=+1\0\-1 for index lead seasonal_data (DJF,MAM,JJA,SON)
    for season_now in range(0,4):
        data_now = season_data[season_now]
        data_size = data_now.shape
        coef_now = np.zeros((1, data_size[1], data_size[2]))
        for lat in range(0, data_size[1]):
            for lon in range(0, data_size[2]):
                if math.isnan(data_now[0, lat, lon]):
                    coef_now[0, lat, lon] = np.nan
                else:
                    if lag[season_now] == 1:
                        index_now = index_data[:-1]
                        data_series = data_now[1:, lat, lon]
                        coef_now[0, lat, lon] = get_LR_coef(data_series, index_now)
                    elif lag[season_now] == 0:
                        index_now = index_data
                        data_series = data_now[:, lat, lon]
                        coef_now[0, lat, lon] = get_LR_coef(data_series, index_now)
                    elif lag[season_now] == -1:
                        index_now = index_data[1:]
                        data_series = data_now[:-1, lat, lon]
                        coef_now[0, lat, lon] = get_LR_coef(data_series, index_now)
        season_coef.append(coef_now)
    return season_coef



