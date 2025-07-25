import xarray as xr
import numpy as np
import os
import subprocess
import datetime as dt
import pandas as pd
import netCDF4 as ncdf

#class eof
#mask_file 用于确定范围
#eof等直接用Dataset储存 最后保存为nc文件
#要画图的时候用netcdf4方法读取

#copy from kido 2021

def convert_longitude_to_360(lon):
    """
    Convert longitude from -180-180 range to 0-360 range.

    Parameters:
    lon (float): Longitude in -180-180 range.

    Returns:
    float: Longitude in 0-360 range.
    """
    if lon < 0:
        return lon + 360
    return lon

def dt64_to_dt(time):
    #将np.datetime64 转变为 dt.datetime
    #https://qiita.com/Kanahiro/items/3ed1372358735c83884c
    #https://stackoverflow.com/questions/22082103/on-windows-how-to-convert-a-timestamps-before-1970-into-something-manageable
    ntime=len(time)
    dt_out=[]
    for i in range(0,ntime):
        time_now = time[i].data
        #date_time64_to_seconds
        scs = int(time_now.astype(dt.datetime) * 1e-9)
        #时间戳windows上不接受负值，只能出此下策了
        dt_now = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=scs)
        dt_out.append(dt_now)
    dt_out=np.asarray(dt_out)
    return(dt_out)

def float_to_dt(float_array, reference_date=dt.datetime(1970, 1, 1)):
    """
    Convert a float array to an array of datetime.datetime objects.

    Parameters:
    float_array (np.ndarray): Array of floats representing time increments.
    reference_date (datetime): The reference datetime. Defaults to Unix epoch time.

    Returns:
    np.ndarray: Array of datetime.datetime objects.
    """
    return np.array([reference_date + dt.timedelta(days=float_day) for float_day in float_array])


def moving_smooth(time_series,window_size):
    # time_series:list data
    # window_size:int
    time_series=time_series.tolist()
    numbers_series = pd.Series(time_series)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1:]
    final_array = np.array(final_list)
    return final_array
    
def select_region_LL_file(ds_in,varname,lat1=-90.0,lat2=90.0,lon1=0.0,lon2=360.0,lonname="lon",latname="lat"):
    #ds = xr.open_dataset(fname)
    ds = ds_in
    lc=ds.coords[lonname]
    la=ds.coords[latname]
    ilc=np.where((lc>=lon1)&(lc<=lon2))[0]
    ila=np.where((la>=lat1)&(la<=lat2))[0]

    if (len(ilc) == 0):
        ilcdis=np.abs(lc-lon1)
        ilcmin=np.min(ilcdis)
        if (len(np.shape(ilcmin))==0):
            ilcmin=np.asarray([ilcmin])
        ilc=np.where(ilcdis==ilcmin[0])[0]
    if (len(ila) == 0):
        iladis=np.abs(la-lat1)
        ilamin=np.min(iladis)
        if (len(np.shape(ilamin))==0):
            ilamin=np.asarray([ilamin])
        ila=np.where(iladis==ilamin[0])[0]
    dict_in={}
    dict_in[latname]=ila
    dict_in[lonname]=ilc

    # Extract a slice of the data
    var = ds[varname].isel(dict_in)
    ds.close()
    return(var)

#作用 选完以后全部变量统一成var格式了
def select_region_TLL_files(ds_in,varname,dt1=dt.datetime(1900,1,1,0,0,0),dt2=dt.datetime(2900,1,1,0,0,0),lat1=-90.0,lat2=90.0,lon1=0.0,lon2=360.0,lonname="lon",latname="lat",timename="time"):
    #nfile=len(fnames)
    ds = ds_in
    var_all=[]
    #for ifile in range(0,nfile):
        #ds = xr.open_dataset(fnames[ifile],decode_times=True)
    lc=ds.coords[lonname]
    la=ds.coords[latname]
    time=ds.coords[timename]
    if (time.dtype=="datetime64[ns]"):
        dt_time=dt64_to_dt(time)
    elif (time.dtype=="float"):
        dt_time = float_to_dt(time.data,reference_date=dt.datetime(1900, 1, 1, 0, 0))
        # dt_time = xr.DataArray(dt_time_data,dims="time")
    else:
        dt_time=time
    ilc=np.where((lc>=lon1)&(lc<=lon2))[0]
    ila=np.where((la>=lat1)&(la<=lat2))[0]
    itime=np.where((dt_time>=dt1)&(dt_time<=dt2))[0]

    if (len(ilc) == 0):
        ilcdis=np.abs(lc-lon1)
        ilcmin=np.min(ilcdis)
        if (len(np.shape(ilcmin))==0):
            ilcmin=np.asarray([ilcmin])
        ilc=np.where(ilcdis==ilcmin[0])[0]
    if (len(ila) == 0):
        iladis=np.abs(la-lat1)
        ilamin=np.min(iladis)
        if (len(np.shape(ilamin))==0):
            ilamin=np.asarray([ilamin])
        ila=np.where(iladis==ilamin[0])[0]

    dict_in={}
    dict_in[timename]=itime
    dict_in[latname]=ila
    dict_in[lonname]=ilc

    # Extract a slice of the data
    var = ds[varname].isel(dict_in)
    if (len(itime)>=1):
        var_all.append(var)
    ds.close()
    var_all = xr.concat(var_all, dim=timename)
    return(var_all)

#def get_mask(ds_in, xr_mask):

def get_weight_eof(ds_in,normalize=False,lonname="lon",latname="lat"):
    var_in=ds_in["var"]
    # Normalize anomalies with area-averaged standard deviation
    if (normalize == True):
        var_std=np.nanstd(var_in,axis=0)
        factor=1.0/np.nanmean(np.nanmean(var_std,axis=1),axis=0)
    # Latitudinal weightening
    lon_wgt=np.ones(len(var_in[lonname]))
    lat_wgt=np.sqrt(np.cos(var_in[latname]*np.pi/180.0))
    wgtx,wgty=np.meshgrid(lon_wgt,lat_wgt)
    wgts=wgtx*wgty*factor*ds_in["mask"]
    ds_in["weight"]=wgts
    return(ds_in)

def set_mask(ds_in):
    var_in=ds_in["var"][:]
    huge=1.0e20
    var_max=np.max(np.abs(var_in),axis=0)
    mask_in=ds_in["mask"]
    var_max_stacked = var_max.stack(z=("lat","lon"))
    mask_in_stacked = mask_in.stack(z=("lat","lon"))
    mask_in_stacked[var_max_stacked.isnull()]=0.0
    mask_in=mask_in_stacked.unstack()
    ds_in["mask"][:]=mask_in[:]
    return(ds_in)

def merge_xarray_TLL(vars_list,lonname="lon",latname="lat",varname="var"):
    # Merge lon-lat data into single xarray data
    nvar=len(vars_list)
    var_merged=[]
    for ivar in range(0,nvar):
        d_1=vars_list[ivar]
        d_1=d_1.rename({'lon': 'lon_'+str(ivar),'lat': 'lat_'+str(ivar)})
        d_1=d_1.rename({varname: varname+'_'+str(ivar)})
        d_1=d_1.rename({'mask': 'mask_'+str(ivar)})
        d_1=d_1.rename({'weight': 'weight_'+str(ivar)})
        var_merged.append(d_1)
    d=xr.merge(var_merged,compat='override')
    d.attrs["nvar"]=nvar
    return(d)

def flatten_TLL(vars_list,timename='time',varname="var",isweight=True):
    # Make data array into (time x lonlat), and apply weight
    nvar=vars_list.attrs["nvar"]
    ndim_all=0
    ntime=len(vars_list[timename])
    for ivar in range(0,nvar):
        a=vars_list["mask_"+str(ivar)]#.stack(z=("lat","lon"))
        ndim_all+=np.sum(a)
    ndim_all=int(ndim_all)
    vec_eof=np.zeros((ntime,ndim_all))
    im1=0;im2=0
    for ivar in range(0,nvar):
        var_tmp=vars_list[varname+"_"+str(ivar)]
        mask_tmp=vars_list["mask_"+str(ivar)]
        weight_tmp=vars_list["weight_"+str(ivar)]
        ndim_tmp=np.shape(var_tmp)
        var_tmp= var_tmp.stack(z=("lat_"+str(ivar),"lon_"+str(ivar)))
        mask_tmp=mask_tmp.stack(z=("lat_"+str(ivar),"lon_"+str(ivar)))
        weight_tmp=weight_tmp.stack(z=("lat_"+str(ivar),"lon_"+str(ivar)))
        ind_valid=np.where(mask_tmp==1.0)
        if (isweight == True):
            var_tmp=var_tmp*weight_tmp
        a=var_tmp[:,ind_valid[0]]
        ndim_tmp=np.shape(a)
        im2=im1+ndim_tmp[1]
        vec_eof[:,im1:im2]=a[:]
        im1=im2

    d_2=xr.DataArray(vec_eof, dims=[timename,'z'],
             coords={timename: vars_list[timename], 'z': np.arange(0,ndim_all)},name=varname+"_flatten")
    var_merged=xr.merge([vars_list,d_2],compat='override')
    return(var_merged)

def revert_TLL(var_vector,vars_list,time):
    timename=time.name
    # Convert TL to geographical (time x lat x lon ) data
    nvar=vars_list.attrs["nvar"]
    return_array=[]
    ndim=np.shape(var_vector)
    ntime=ndim[0]
    im1=0
    im2=0
    var_return=[]
    for ivar in range(0,nvar):
        mask_tmp=vars_list["mask_"+str(ivar)]
        mask_tmp_stacked=mask_tmp.stack(z=("lat_"+str(ivar),"lon_"+str(ivar)))
        ndim_tmp=np.shape(mask_tmp)
        tmp_out=np.ones((ntime,ndim_tmp[0],ndim_tmp[1]))*np.nan
        ind_valid=np.where(mask_tmp==1.0)
        im2=im1+int(np.sum(mask_tmp))
        tmp_out[:,ind_valid[0],ind_valid[1]]=var_vector[:,im1:im2]
        im1=im2
        d_out=xr.DataArray(tmp_out, dims=[timename,'lat_'+str(ivar),'lon_'+str(ivar)],
             coords={timename: time,"lat_"+str(ivar):vars_list["lat_"+str(ivar)]
             ,"lon_"+str(ivar):vars_list["lon_"+str(ivar)]},name="var_"+str(ivar))
        var_return.append(d_out)
    var_return=xr.merge(var_return)
    return(var_return)

def get_eof(nmode,var_in):
    nvar=var_in.attrs["nvar"]
    modes=np.arange(1,nmode+1,1)
    var_eof=var_in["var_flatten"].data
    ndim=np.shape(var_eof)
    ntime=ndim[0]
    nx=ndim[1]
    eof_s=True
    if (nx < ntime):
        eof_s=False
    if (eof_s == True):
        # Eigenanalysis for time dimension
        cov_t=np.dot(var_eof,np.transpose(var_eof))/nx
        w, v = np.linalg.eig(cov_t)
        variance=100*w[0:nmode]/np.sum(w)
        pc_ori=v[:,0:nmode]
        vec_eof=np.transpose(np.dot(np.transpose(pc_ori),var_eof)) # NX x nmode
        for imode in range(0,nmode):
            dnorm=np.dot(np.transpose(vec_eof[:,imode]),vec_eof[:,imode])
            vec_eof[:,imode]=vec_eof[:,imode]/np.sqrt(dnorm)
    else:
        # Eigenanalysis for space dimension
        cov_s=np.dot(np.transpose(var_eof),var_eof)/ntime
        w, v = np.linalg.eig(cov_s)
        variance=100*w[0:nmode]/np.sum(w)
        vec_eof=v[:,0:nmode]
        for imode in range(0,nmode):
            dnorm=np.dot(np.transpose(vec_eof[:,imode]),vec_eof[:,imode])
            vec_eof[:,imode]=vec_eof[:,imode]/np.sqrt(dnorm)
    pc=np.dot(var_eof,vec_eof)
    pc=np.real(pc)
    variance=np.real(variance)
    mode=xr.DataArray(np.asarray(np.arange(1,nmode+1,1),dtype="float64"),dims=["mode"],name="mode")
    eof_spatial=revert_TLL(np.transpose(vec_eof),var_in,mode)
    variance_out=xr.DataArray(variance, dims=['mode'],
             coords={'mode':mode},name="variance")

    var_all=[]
    var_all.append(variance_out)
    var_all.append(eof_spatial)
    for ivar in range(0,nvar):
        pc_out=xr.DataArray(pc, dims=['time','mode'],
             coords={'time': var_in["time"],'mode':mode},name="pc_"+str(ivar))
        var_all.append(pc_out)
        var_all.append(var_in["mask_"+str(ivar)])
        var_all.append(var_in["weight_"+str(ivar)])
    var_ret=xr.merge(var_all)
    var_ret.attrs["nvar"]=nvar
    return(var_ret)

def eof_to_ncdf(var_in,fnames_out,time_units="days since 1900-01-01 00:00:00"):
    nvar=var_in.attrs["nvar"]
    for ivar in range(0,nvar):
        var_tmp=[]
        var_tmp.append(var_in["variance"])
        var_tmp.append(var_in["pc_"+str(ivar)])
        var_tmp.append(var_in["mask_"+str(ivar)])
        var_tmp.append(var_in["weight_"+str(ivar)])
        var_tmp.append(var_in["var_"+str(ivar)])
        weight_ori=np.copy(var_in["weight_"+str(ivar)].data)
        zweight=var_in["weight_"+str(ivar)]
        tmp=var_in["weight_"+str(ivar)].data
        tmp[tmp==0.0]=np.nan
        zweight.data=1/tmp
        #zweight[var_in["weight_"+str(ivar)]==0.0]=np.nan
        zweight=zweight.to_dataset()
        zweight=zweight.rename({'weight_'+str(ivar): 'zweight'})
        var_tmp.append(zweight)
        var_ret=xr.merge(var_tmp)
        var_ret["weight_"+str(ivar)].data=weight_ori
        var_ret=var_ret.rename({'lon_'+str(ivar): 'lon'})
        var_ret=var_ret.rename({'lat_'+str(ivar): 'lat'})
        var_ret=var_ret.rename({'pc_'+str(ivar): 'pc'})
        var_ret=var_ret.rename({'mask_'+str(ivar): 'mask'})
        var_ret=var_ret.rename({'weight_'+str(ivar): 'weight'})
        var_ret=var_ret.rename({'var_'+str(ivar): 'eof_vec'})
        #if (os.path.exists(fnames_out[ivar])==True):
            #command="rm "+fnames_out[ivar]
            #subprocess.call(command.split())
        var_ret.to_netcdf(fnames_out[ivar],encoding={'time':{'dtype': 'float64','units':time_units}})

#from ds_eof to ncfile
def do_eof(ds_in, xr_mask_in, nmode, fnames_out, varname=""):
    ds_eof = ds_in
    #mask xr2data
    mask_in=xr_mask_in.data
    mask_in[mask_in>0] = 1.0
    if varname=="":
        var_in = ds_in["var"]
    else:
        var_in = ds_in[varname]
    a = np.max(var_in.data, axis=0)

    #get mask into ds
    eof_all = []
    mask_in[np.isnan(a)] = 0.0
    ds_eof['mask'] = (("lat", "lon"), mask_in)
    ds_eof = get_weight_eof(ds_eof, normalize=True)
    ds_eof = set_mask(ds_eof)
    eof_all.append(ds_eof)

    # Merge all variables
    var_merged = merge_xarray_TLL(eof_all)
    # Flatten variable to perform EOF
    var_merged = flatten_TLL(var_merged)
    # Do EOF analysis
    a = get_eof(nmode, var_merged)
    eof_to_ncdf(a, fnames_out)
    print("eof result has been stored in " + fnames_out[0])

def project_pc(fnames_vec, fnames_var, var_name, fnames_out,dt1=dt.datetime(1900,1,1,0,0,0),dt2=dt.datetime(2900,1,1,0,0,0)):
    # ds_vec: eof数据 ds_var: 原始数据
    # fnames_vec =之前输出的eof文件 如果用了滑动平均该少两个月？
    # fnames_var = 原始的没有去除滑动平均的文件
    nfile = len(fnames_vec)
    for ifile in range(0, nfile):
        fnames_vec_tmp = fnames_vec[ifile]
        fnames_var_tmp = fnames_var[ifile]
        nfile_tmp = len(fnames_var_tmp)
        # Projecting variable
        ds_vec = []
        ds_var = []
        for i in range(0, nfile_tmp):
            ds_in = xr.open_dataset(fnames_vec_tmp[i])
            lon = ds_in["lon"]
            lat = ds_in["lat"]
            dt_data = ds_in["time"]
            if (dt_data.dtype == "datetime64[ns]"):
                dt_data = dt64_to_dt(dt_data)
            lon1 = np.min(lon)
            lon2 = np.max(lon)
            lat1 = np.min(lat)
            lat2 = np.max(lat)
            dt_data_1 = dt_data
            ds_var0 = xr.open_dataset(fnames_var_tmp[i])

            ds_var_tmp = select_region_TLL_files(ds_var0, var_name, dt1=dt1, dt2=dt2, lat1=lat1, lat2=lat2,
                                                 lon1=lon1, lon2=lon2, lonname="lon", latname="lat", timename="time")
            ds_var_tmp = ds_var_tmp.to_dataset()
            ds_var_tmp['mask'] = ds_in['mask']
            ds_var_tmp['weight'] = ds_in['weight']
            ds_vec.append(ds_in)
            ds_var.append(ds_var_tmp)
            ds_in.close()
        vec_merged = merge_xarray_TLL(ds_vec, varname="eof_vec")
        vec_merged = flatten_TLL(vec_merged, varname="eof_vec", timename="mode", isweight=False)

        var_merged = merge_xarray_TLL(ds_var, varname=var_name)
        var_merged = flatten_TLL(var_merged, varname=var_name)
        # Obtain PCs
        pc_out = np.dot(var_merged[var_name + "_flatten"].data, np.transpose(vec_merged["eof_vec_flatten"].data))
        pc_ori = ds_in["pc"]
        data = xr.DataArray(pc_out, dims=["time", "mode"],
                            coords={"time": var_merged["time"], "mode": vec_merged["mode"]}, name="pc")
        data = data.to_dataset()
        data["variance"] = vec_merged["variance"]
        # if (os.path.exists(fnames_out) == True):
        #     command = "rm " + fnames_out
        #     subprocess.call(command.split())
        time_units = "days since " + str(dt.datetime(1900, 1, 1, 0, 0, 0))
        data.to_netcdf(fnames_out[ifile], encoding={'time': {'dtype': 'float64', 'units': time_units}})