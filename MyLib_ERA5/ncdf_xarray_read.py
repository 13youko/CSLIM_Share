import xarray as xr
import datetime as dt
import numpy as np
def dt64_to_dt(time):
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
def select_region_LL_file(fname,varname,lat1=-90.0,lat2=90.0,lon1=0.0,lon2=360.0,lonname="lon",latname="lat"):
    ds = xr.open_dataset(fname)
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

def select_region_TLL_files(fnames,varname,dt1=dt.datetime(1900,1,1,0,0,0),dt2=dt.datetime(2900,1,1,0,0,0),lat1=-90.0,lat2=90.0,lon1=0.0,lon2=360.0,lonname="lon",latname="lat",timename="time"):
    nfile=len(fnames)
    var_all=[]
    for ifile in range(0,nfile):
        ds = xr.open_dataset(fnames[ifile],decode_times=True)
        lc=ds.coords[lonname]
        la=ds.coords[latname]
        time=ds.coords[timename]
        if (time.dtype=="datetime64[ns]"):
            dt_time=dt64_to_dt(time)
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
def select_region_TLLL_files(fnames,varname,dt1=dt.datetime(1900,1,1,0,0,0),dt2=dt.datetime(2900,1,1,0,0,0),lev1=0.0,lev2=10000.0,\
    lat1=-90.0,lat2=90.0,lon1=0.0,lon2=360.0,lonname="lon",latname="lat",levname="lev",timename="time"):
    nfile=len(fnames)
    var_all=[]
    for ifile in range(0,nfile):
        ds = xr.open_dataset(fnames[ifile],decode_times=True)
        lc=ds.coords[lonname]
        la=ds.coords[latname]
        lv=ds.coords[levname]
        time=ds.coords[timename]
        if (time.dtype=="datetime64[ns]"):
            dt_time=dt64_to_dt(time)
        else:
            dt_time=time

        ilc=np.where((lc>=lon1)&(lc<=lon2))[0]
        ila=np.where((la>=lat1)&(la<=lat2))[0]
        ilv=np.where((lv>=lev1)&(lv<=lev2))[0]
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
        if (len(ilv) == 0):
            ilvdis=np.abs(lv-lev1)
            ilvmin=np.min(ilvdis)
            if (len(np.shape(ilvmin))==0):
                ilvmin=np.asarray([ilvmin])            
            ilv=np.where(ilvdis==ilvmin[0])[0]

        dict_in={}
        dict_in[timename]=itime
        dict_in[levname]=ilv
        dict_in[latname]=ila
        dict_in[lonname]=ilc

        # Extract a slice of the data
        var = ds[varname].isel(dict_in)
        if (len(itime)>=1):
            var_all.append(var)
        ds.close()
    var_all = xr.concat(var_all, dim=timename)

    return(var_all)

