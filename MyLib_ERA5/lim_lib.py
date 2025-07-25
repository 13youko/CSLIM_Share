# LIM script
import netCDF4 as ncdf
import numpy as np
import datetime as dt
import numpy.linalg as LA
from scipy import signal
from dateutil.relativedelta import relativedelta
import calendar

class pcs_LIM:
    def __init__(self,fnames_in,modes,time,pcs,percent):
        self.fnames_in=fnames_in
        self.modes=modes
        self.time=time
        self.pcs=pcs
        self.percent=percent
class oper_LIM:
    def __init__(self,B_real,B_imag,Q_real,Q_imag,C_0,modes,time,tau):
        self.units="1/month"
        self.B_real=B_real
        self.Q_real=Q_real
        self.B_imag=B_imag
        self.Q_imag=Q_imag
        self.C_0=C_0
        self.modes=modes
        self.time=time
        self.tau=tau
        self.fname_oripc=""
        self.fname_matrix=""
    def add_C_tau(self,C_tau):
        self.C_tau=C_tau
    def add_pcname(self,fname_pc):
        self.fname_oripc=fname_pc
    def add_matrixname(self,fname_matrix):
        self.fname_matrix=fname_matrix
    def get_B(self):
        self.B=self.B_real+self.B_imag*1j
    def get_Q(self):
        self.Q=self.Q_real+self.Q_imag*1j

def get_statevectors_number(fnames_in,modes,dt1=dt.datetime(1900,1,1,0,0,0),dt2=dt.datetime(2900,12,31,0,0,0),timename='time',modename='mode',valname="eigval"):
    """ Get state vectors from EOF data, with specifying the truncations numbe
    Input:
    fnames_in: Name of input EOF files
    fname_out: Name of output file
    modes: Number of PCs
    """
    nfile=len(fnames_in)
    nmode=np.sum(modes) # Dimension of state vector
    ifile=0
    nc_in=ncdf.Dataset(fnames_in[ifile],'r')
    time=nc_in.variables[timename][:]
    timeunits=nc_in.variables[timename].units
    cal=ncdf.num2date(time,timeunits,calendar="standard")
    ical=np.where((cal >= dt1) & (cal <= dt2))[0]
    nc_in.close()

    ntime=len(time[ical])
    # Prepare array
    data_in=np.zeros((ntime,nmode))
    mode=np.linspace(1.0,nmode*1.0,nmode)
    percent=np.zeros(nfile)
    it1=0;it2=0
    varname_pc="pc"
    for ifile in range(0,nfile):
        nc_in=ncdf.Dataset(fnames_in[ifile],'r')
        time=nc_in.variables[timename][:]
        timeunits=nc_in.variables[timename].units
        cal=ncdf.num2date(time,timeunits,calendar="standard")
        ical=np.where((cal >= dt1) & (cal <= dt2))[0]
        var=nc_in.variables[varname_pc][ical,:]
        it1=it2
        it2=it1+modes[ifile]
        data_in[:,it1:it2]=var[:,0:(it2-it1)]
        exp=nc_in.variables[valname][0:(it2-it1)]
        percent[ifile]=np.sum(exp)
        nc_in.close()
    pcs=pcs_LIM(fnames_in,modes,cal[ical],data_in,percent)
    return(pcs)

def get_statevectors_percent(fnames_in,exp_percent,dt1=dt.datetime(1900,1,1,0,0,0),dt2=dt.datetime(2900,12,31,0,0,0),timename='time',modename='mode',valname="variance"):
    """ Get state vectors from EOF data, with specifying the truncations numbe
    Input:
    fnames_in: Name of input EOF files
    exp_percent: Number of percent (Should be a list with
           dimension of len(fnames_in)
    """
    nfile=len(fnames_in)
    modes=[]
    for ifile in range(0,nfile):
        nc_in=ncdf.Dataset(fnames_in[ifile],'r')
        exp=nc_in.variables[valname][:]
        nmodes=len(exp)
        percent=np.zeros(nmodes)
        # Get the numbers of truncated EOFs
        for im in range(0,nmodes):
            percent[im]=np.sum(exp[0:im+1])
        index_int=np.where(percent>=exp_percent[ifile])[0]
        modes.append(index_int[0]+1)

    nmode=np.sum(modes) # Dimension of state vector
    ifile=0
    nc_in=ncdf.Dataset(fnames_in[ifile],'r')
    time=nc_in.variables[timename][:]
    timeunits=nc_in.variables[timename].units
    cal=ncdf.num2date(time,timeunits,calendar="standard")
    ical=np.where((cal >= dt1) & (cal <= dt2))[0]
    nc_in.close()

    ntime=len(time[ical])
    # Prepare array
    data_in=np.zeros((ntime,nmode))
    mode=np.linspace(1.0,nmode*1.0,nmode)
    percent=np.zeros(nfile)
    it1=0;it2=0
    varname_pc="pc"
    for ifile in range(0,nfile):
        nc_in=ncdf.Dataset(fnames_in[ifile],'r')
        time=nc_in.variables[timename][:]
        timeunits=nc_in.variables[timename].units
        cal=ncdf.num2date(time,timeunits,calendar="standard")
        ical=np.where((cal >= dt1) & (cal <= dt2))[0]
        var=nc_in.variables[varname_pc][ical,:]
        it1=it2
        it2=it1+modes[ifile]
        data_in[:,it1:it2]=var[:,0:(it2-it1)]
        exp=nc_in.variables[valname][0:(it2-it1)]
        percent[ifile]=np.sum(exp)
        nc_in.close()
    pcs=pcs_LIM(fnames_in,modes,cal[ical],data_in,percent)
    return(pcs)

def save_pcs_ncdf(fname_pc, pc_in, timename='time', modename='mode', ref_dt=dt.datetime(1900, 1, 1, 0, 0, 0),
                  varname_pc="pc"):
    timeunits = "days since " + str(ref_dt)
    cals = np.copy(pc_in.time)
    time_out = np.zeros((len(cals)))
    for it in range(0, len(cals)):
        tmp = cals[it] - ref_dt
        time_out[it] = tmp.days + (tmp.seconds) / (60 * 60 * 24)

    nmode = np.sum(pc_in.modes)
    mode_out = np.arange(1.0, nmode + 1, 1)
    nc_out = ncdf.Dataset(fname_pc, "w")
    nc_out.createDimension(timename, None)
    nc_out.createDimension(modename, nmode)
    nc_out.createVariable(timename, time_out.dtype, (timename))
    nc_out.variables[timename].units = timeunits
    nc_out.createVariable(modename, mode_out.dtype, (modename))
    nc_out.variables[modename].units = ""
    pc_out = nc_out.createVariable(varname_pc, pc_in.pcs.dtype, (timename, modename))
    nc_out.variables[timename][:] = time_out
    nc_out.variables[modename][:] = mode_out
    nc_out.variables[varname_pc][:] = pc_in.pcs
    nc_out.file_names = pc_in.fnames_in
    nc_out.modes = pc_in.modes
    nc_out.variance = pc_in.percent
    nc_out.close()


def read_pcs_ncdf(fname_pc, varname_pc="pc"):
    nc_in = ncdf.Dataset(fname_pc, 'r')
    mode = nc_in.variables['mode'][:]
    modes = nc_in.getncattr('modes')
    fnames_in = nc_in.getncattr('file_names')
    percent = nc_in.getncattr('variance')
    time = nc_in.variables['time'][:]
    timeunits = nc_in.variables["time"].units
    cal = ncdf.num2date(time, timeunits, calendar="standard")
    eof_pc = nc_in.variables[varname_pc][:, :]
    nc_in.close()
    pcs = pcs_LIM(fnames_in, modes, cal, eof_pc, percent)
    return (pcs)

def get_covariance(data1,data2):
    data1=data1-np.mean(data1,axis=0)
    data2=data2-np.mean(data2,axis=0)
    C_0=np.dot(np.transpose(data1),data1)/len(data1)
    C_tau=np.dot(np.transpose(data2),data1)/len(data1)
    return(C_0,C_tau)
###
# Constant 部分

def get_covar_constant(pcs,tau,time_units="1/month",time_scale=1.0):
    modes=pcs.modes
    nmode=np.sum(modes)
    time=pcs.time
    ntime=np.shape(time)[0]
    eof_pc=pcs.pcs
    data1=eof_pc[0:(ntime-1-tau),:]
    data2=eof_pc[tau:(ntime-1),:]
    C_0,C_tau=get_covariance(data1,data2)
    C_0_array=np.zeros((1,nmode,nmode))
    C_tau_array=np.zeros((1,nmode,nmode))
    C_0_array[0,:,:]=np.copy(C_0)
    C_tau_array[0,:,:]=np.copy(C_tau)
    return(C_0_array,C_tau_array)

def get_covar_aveseason(pcs,tau,time_scale=1,time_units="1/month"):
    tau_days=tau*time_scale
    time_out=np.zeros((12))
    modes=pcs.modes
    nmode=np.sum(modes)
    time=pcs.time
    ntime=np.shape(time)[0]
    eof_pc=pcs.pcs
    # 12
    C_0_array=np.zeros((12,nmode,nmode))
    C_tau_array=np.zeros((12,nmode,nmode))
    for im in range(0,12):
        im_bef=im-1
        if (im ==0):
            im_bef=11
        else:
            im_bef=im-1
        if (im ==11):
            im_aft=0
        else:
            im_aft=im+1
        im_cnt=im

       # C0_bef
        data1=np.copy(eof_pc[im_bef:(ntime-1-tau):12,:])
        data2=np.copy(eof_pc[im_bef+tau:(ntime-1):12,:])
        C_0_bef,C_tau_bef=get_covariance(data1,data2)
        # C0_cent
        data1=np.copy(eof_pc[im_cnt:(ntime-1-tau):12,:])
        data2=np.copy(eof_pc[im_cnt+tau:(ntime-1):12,:])
        C_0_cnt,C_tau_cnt=get_covariance(data1,data2)
        # C0_aft
        data1=np.copy(eof_pc[im_aft:(ntime-1-tau):12,:])
        data2=np.copy(eof_pc[im_aft+tau:(ntime-1):12,:])
        C_0_aft,C_tau_aft=get_covariance(data1,data2)
        C_0=(C_0_bef+C_0_cnt+C_0_aft)/3
        C_tau=(C_tau_bef+C_tau_cnt+C_tau_aft)/3
        C_0_array[im,:,:]=C_0
        C_tau_array[im,:,:]=C_tau
    return(C_0_array,C_tau_array)

def C_to_opr_constant(C_0,C_tau,modes,tau,time_units="1/month",time_scale=1.0):
    nmode=np.sum(modes)
    # C_0_tmp=C_0[0,:,:]
    # C_tau_tmp=C_tau[0,:,:]
    C_0_tmp=C_0[:,:]
    C_tau_tmp=C_tau[:,:]
    Gt=np.dot(C_tau_tmp,LA.inv(C_0_tmp))
    Gtw,Gtv = LA.eig(Gt)
    tau_days=tau*time_scale
    Lt=(1.0/tau_days)*np.log(Gtw)
    Gtinv=LA.inv(Gtv)
    # Operator
    B_real=np.zeros((1,nmode,nmode))
    B_imag=np.zeros((1,nmode,nmode))
    Q_real=np.zeros((1,nmode,nmode))
    Q_imag=np.zeros((1,nmode,nmode))
    B_tmp=np.dot(np.dot(Gtv,np.diag(Lt)),Gtinv)
    Q_tmp=-np.dot(B_tmp,C_0_tmp)-np.dot(C_0_tmp,np.conjugate(np.transpose(B_tmp)))
    B_real[0,:,:]=np.copy(np.real(B_tmp))
    B_imag[0,:,:]=np.copy(np.imag(B_tmp))
    Q_real[0,:,:]=np.copy(np.real(Q_tmp))
    Q_imag[0,:,:]=np.copy(np.imag(Q_tmp))
    time=np.zeros((1))
    opr=oper_LIM(B_real,B_imag,Q_real,Q_imag,C_0,modes,time,tau)
    opr.add_C_tau(C_tau)
    opr.units=time_units
    return(opr)

def get_LIMoper_constant(pcs,tau,time_units="1/month",time_scale=1.0):
    modes=pcs.modes
    nmode=np.sum(modes)
    time=pcs.time
    ntime=np.shape(time)[0]
    eof_pc=pcs.pcs
    data1=eof_pc[0:(ntime-1-tau),:]
    data2=eof_pc[tau:(ntime-1),:]
    C_0,C_tau=get_covariance(data1,data2)
    opr=C_to_opr_constant(C_0,C_tau,modes,tau,time_units,time_scale)
    return(opr)
###

###
#aveseason部分
def C_to_opr_aveseason(C_0,C_tau,modes,tau,time_units="1/month",time_scale=1.0):
    tau_days=tau*time_scale
    time_out=np.zeros((12))
    nmode=np.sum(modes)
    B_real=np.zeros((12,nmode,nmode)); B_imag=np.zeros((12,nmode,nmode))
    Q_real=np.zeros((12,nmode,nmode)); Q_imag=np.zeros((12,nmode,nmode))
    for im in range(0,12):
        Gt=np.dot(C_tau[im,:,:],LA.inv(C_0[im,:,:]))
        Gtw,Gtv = LA.eig(Gt)
        Lt=(1.0/tau_days)*np.log(Gtw)
        Gtinv=LA.inv(Gtv)
        # Operator
        B=np.dot(np.dot(Gtv,np.diag(Lt)),Gtinv)
        B_real[im,:,:]=np.real(B)
        B_imag[im,:,:]=np.imag(B)
    # Q
    for im in range(0,12):
        if (im == 11):
            im_next=0
        else:
            im_next=im+1
        if (im == 0):
            im_bef=11
        else:
            im_bef=im-1
        B=B_real[im,:,:]+B_imag[im,:,:]*1j
        Q=0.5*(C_0[im_next,:,:]-C_0[im_bef,:,:])-np.dot(B,C_0[im,:,:])-np.dot(C_0[im,:,:],np.conjugate(np.transpose(B)))
        Q_real[im,:,:]=np.real(Q);Q_imag[im,:,:]=np.imag(Q)
    opr=oper_LIM(B_real,B_imag,Q_real,Q_imag,C_0,modes,time_out,tau)
    opr.add_C_tau(C_tau)
    opr.units=time_units
    return(opr)

def get_LIMoper_aveseason(pcs,tau,time_scale=1,time_units="1/month"):
    tau_days=tau*time_scale
    time_out=np.zeros((12))
    modes=pcs.modes
    nmode=np.sum(modes)
    time=pcs.time
    ntime=np.shape(time)[0]
    eof_pc=pcs.pcs
    # 12
    C_0_array=np.zeros((12,nmode,nmode))
    C_tau_array=np.zeros((12,nmode,nmode))
    for im in range(0,12):
        im_bef=im-1
        if (im ==0):
            im_bef=11
        else:
            im_bef=im-1
        if (im ==11):
            im_aft=0
        else:
            im_aft=im+1
        im_cnt=im

       # C0_bef
        data1=np.copy(eof_pc[im_bef:(ntime-1-tau):12,:])
        data2=np.copy(eof_pc[im_bef+tau:(ntime-1):12,:])
        C_0_bef,C_tau_bef=get_covariance(data1,data2)
        # C0_cent
        data1=np.copy(eof_pc[im_cnt:(ntime-1-tau):12,:])
        data2=np.copy(eof_pc[im_cnt+tau:(ntime-1):12,:])
        C_0_cnt,C_tau_cnt=get_covariance(data1,data2)
        # C0_aft
        data1=np.copy(eof_pc[im_aft:(ntime-1-tau):12,:])
        data2=np.copy(eof_pc[im_aft+tau:(ntime-1):12,:])
        C_0_aft,C_tau_aft=get_covariance(data1,data2)
        C_0=(C_0_bef+C_0_cnt+C_0_aft)/3
        C_tau=(C_tau_bef+C_tau_cnt+C_tau_aft)/3
        C_0_array[im,:,:]=C_0
        C_tau_array[im,:,:]=C_tau
    opr=C_to_opr_aveseason(C_0_array,C_tau_array,modes,tau,time_units,time_scale)
    return(opr)

def save_operator_ncdf(fname_matrix,oper):
    time=oper.time
    modes=oper.modes
    nmode=np.sum(modes)
    ntime=len(time)
    mode=np.linspace(1.0,nmode*1.0,nmode)
    nc_out=ncdf.Dataset(fname_matrix,"w")
    nc_out.createDimension('time',ntime)
    nc_out.createDimension('mode1',nmode)
    nc_out.createDimension('mode2',nmode)
    nc_out.createVariable('mode1',mode.dtype,('mode1'))
    nc_out.createVariable('mode2',mode.dtype,('mode2'))
    B_real=oper.B_real;B_imag=oper.B_imag
    Q_real=oper.Q_real;Q_imag=oper.Q_imag
    nc_out.createVariable('B_real',B_real.dtype,('time','mode2','mode1'))
    nc_out.createVariable('Q_real',Q_real.dtype,('time','mode2','mode1'))
    nc_out.createVariable('B_imag',B_imag.dtype,('time','mode2','mode1'))
    nc_out.createVariable('Q_imag',Q_imag.dtype,('time','mode2','mode1'))
    nc_out.createVariable('C_0',oper.C_0.dtype,('time','mode2','mode1'))
    nc_out.createVariable('time',time.dtype,('time'))
    units=oper.units
    nc_out.variables['B_real'].units=units
    nc_out.variables['mode1'][:]=mode[:]
    nc_out.variables['mode2'][:]=mode[:]
    nc_out.variables['time'][:]=time[:]
    nc_out.variables['C_0'][:]=oper.C_0[:,:]
    nc_out.variables['B_real'][:]=B_real[:,:]
    nc_out.variables['B_imag'][:]=B_imag[:,:]
    nc_out.variables['Q_real'][:]=Q_real[:,:]
    nc_out.variables['Q_imag'][:]=Q_imag[:,:]
    nc_out.lag=oper.tau
    nc_out.modes=oper.modes
    if (oper.fname_oripc !=""):
        nc_out.fname_pc=fname_matrix
    nc_out.close()

def read_operator_ncdf(fname_matrix):
    nc_matrix = ncdf.Dataset(fname_matrix, 'r')
    B_real = nc_matrix.variables['B_real'][:, :]
    B_imag = nc_matrix.variables['B_imag'][:, :]
    Q_real = nc_matrix.variables['Q_real'][:, :]
    Q_imag = nc_matrix.variables['Q_imag'][:, :]
    time = nc_matrix.variables['time'][:]
    C_0 = nc_matrix.variables['C_0'][:, :]
    tau = nc_matrix.lag
    modes = nc_matrix.modes
    if (np.size(modes) == 1):
        modes = np.asarray([modes])
    opr = oper_LIM(B_real, B_imag, Q_real, Q_imag, C_0, modes, time, tau)
    opr.add_matrixname(fname_matrix)
    nc_matrix.close()
    return (opr)

def integ_LIM_stochastic_forward(oper, fname_ini, fname_out, tdelta, start_dt, end_dt,
                                 store_int, ref_dt=dt.datetime(1900, 1, 1, 0, 0, 0), ind_init=0,
                                 tdelta_unit="month", store_unit="month",
                                 rmnoise=False, runlog=False):
    if (rmnoise == True):#是remove noise的意思，我服了，给哥们吓一跳
        noise_factor = 0.0
    else:
        noise_factor = 1.0

    # Time setting
    # Time stepping
    if (tdelta_unit == 'month'):
        nstep_int = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    if (tdelta_unit == 'day'):
        nstep_int = (end_dt - start_dt).days
    nstep = int(nstep_int / tdelta)

    # Store variables
    if (store_unit == 'month'):
        nstore_int = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    if (store_unit == 'day'):
        nstore_int = (end_dt - start_dt).days
    nstep_store = int(nstore_int / store_int)
    ref_units = 'days since ' + str(ref_dt)
    # state=np.asarray(state)
    # time=np.zeros(nstore)
    # for it in range(0,nstore):
    #     dt_tmp=start_dt+relativedelta(months=(it))+dt.timedelta(days=15)
    #     a=(dt_tmp-ref_dt)
    #     time[it]=a.days
    time_store = []
    time_out = []
    for it in range(0, nstep_store):
        if (store_unit == 'day'):
            tmp = start_dt + relativedelta(days=((it + 1) * store_int))
            time_store.append(tmp)
            tmp1 = tmp - relativedelta(days=(store_int))
            ndays = (tmp - tmp1).days
            tmp1 = tmp1 + relativedelta(days=ndays * 0.5)
            time_out.append(tmp1)
        elif (store_unit == 'month'):
            tmp = start_dt + relativedelta(months=((it + 1) * store_int))
            time_store.append(tmp)
            tmp1 = tmp - relativedelta(months=(store_int))
            ndays = (tmp - tmp1).days
            tmp1 = tmp1 + relativedelta(days=ndays * 0.5)
            time_out.append(tmp1)
    time_store = np.asarray(time_store)
    time_out = np.asarray(time_out)
    nstore = len(time_out)
    time = np.zeros(nstore)
    for it in range(0, nstore):
        a = (time_store[it] - ref_dt)
        time[it] = a.days + (a.seconds) / (60 * 60 * 24)

    # Read matrix
    B_real_in = np.copy(oper.B_real)
    B_imag_in = np.copy(oper.B_imag)
    Q_real_in = np.copy(oper.Q_real)
    Q_imag_in = np.copy(oper.Q_imag)
    B_real_in = np.copy(oper.B_real)
    modes = oper.modes
    ndim = np.shape(B_real_in)
    nmode = ndim[1]
    mode = np.linspace(1.00, 1.0 * nmode, nmode)
    B = B_real_in + B_imag_in * 1j
    ndim = np.shape(B)
    Q = Q_real_in + Q_imag_in * 1j
    ndim_q = np.shape(Q);
    if (len(ndim_q) > 2):
        ncycle = ndim_q[0]
    else:
        ncycle = 1
        Q = np.reshape(Q, (1, nmode, nmode))
        B = np.reshape(B, (1, nmode, nmode))
    Q_evects_array = np.zeros((ncycle, nmode, nmode))
    Q_evals_array = np.zeros((ncycle, nmode))

    for im in range(0, ncycle):
        q_evals, q_evects = LA.eigh(Q[im, :, :])
        sort_idx = q_evals.argsort()
        q_evals = q_evals[sort_idx][::-1]
        q_evects = q_evects[:, sort_idx][:, ::-1]
        num_neg = (q_evals < 0).sum()

        # Correct q
        if num_neg > 0:
            pos_q_evals = q_evals[q_evals > 0]
            scale_factor = q_evals.sum() / pos_q_evals.sum()
            q_evals = q_evals[:-num_neg] * scale_factor
            q_evects = q_evects[:, :-num_neg]
        q_evects = np.array(q_evects)
        num_evals = q_evals.shape[0]
        Q_evects_array[im, :, 0:num_evals] = q_evects
        Q_evals_array[im, 0:num_evals] = q_evals

    # Read initial conditions
    if (fname_ini == ""):
        state = np.zeros((nmode))
        if (start_dt == ""):
            start_dt = ref_dt
    else:
        print("Initial=" + fname_ini)
        nc_ini = ncdf.Dataset(fname_ini, "r")
        time_in = nc_ini.variables["time"][:]
        time_units = nc_ini.variables["time"].units
        try:
            time_cal = nc_ini.variables["time"].calendar
        except AttributeError:
            time_cal = 'standard'
        dt_in = ncdf.num2date(time_in, units=time_units, calendar=time_cal)
        state = nc_ini.variables["pc"][ind_init, :]
        nc_ini.close()
        start_yymmdd = dt_in[ind_init]
        start_dt = dt.datetime(start_yymmdd.year, start_yymmdd.month, start_yymmdd.day, start_yymmdd.hour,
                               start_yymmdd.minute, start_yymmdd.second)
    state = np.reshape(state, (nmode))
    state = np.asarray(state)
    print("Output=" + fname_out)
    nc_out = ncdf.Dataset(fname_out, "w")
    nc_out.createDimension('mode', nmode)
    nc_out.createDimension('time', nstore)
    nc_out.createVariable('mode', mode.dtype, ('mode'))
    nc_out.createVariable('time', mode.dtype, ('time'))
    nc_out.createVariable('pc', mode.dtype, ('time', 'mode'))
    nc_out.variables['mode'][:] = mode[:]
    nc_out.variables['time'][:] = time[:]
    nc_out.variables['time'].units = ref_units
    nc_out.modes = modes

    istore = 0
    avg_state = np.zeros(nmode)
    avg_state[:] = 0.0
    istep_store = 0
    Gt = np.identity(nmode)
    init_state = np.copy(state)
    init_month = start_dt.month
    state = init_state
    avg_state = np.copy(init_state)
    istep_store = 1;
    istore = 0
    for it in range(1, nstep + 1):
        itime = it * tdelta
        imonth = int(itime)
        if (tdelta_unit == 'month'):
            dt_now = start_dt + relativedelta(months=int(itime))
            dofmonth = calendar.monthrange(dt_now.year, dt_now.month)[1]
            dt_now = dt_now + dt.timedelta(days=(itime - imonth) * dofmonth)
        else:
            dt_now = start_dt + relativedelta(days=int(itime))

        # print(dt_now)
        month_ind = dt_now.month - 1
        imonth_f = int(itime) % ncycle
        B_tmp = B[imonth_f, :, :]
        deterministic = np.dot(B_tmp, state) * tdelta
        random = np.random.normal(size=(nmode))
        stochastic = np.dot(Q_evects_array[imonth_f, :, :], (np.sqrt(Q_evals_array[im, :] * tdelta) * random))
        # Integrate the LIM using the Heun method
        update_1 = deterministic + stochastic * noise_factor
        state_tmp = state + update_1
        deterministic = np.dot(B_tmp, state_tmp) * tdelta
        update_2 = deterministic + stochastic * noise_factor
        state_tmp = state + 0.5 * (update_1 + update_2)
        state = state_tmp

        istep_store = istep_store + 1
        avg_state = avg_state + state
        if (np.isnan(np.mean(avg_state))):
            print("f")
            print("ff")

        if (dt_now >= time_store[istore]):
            if (runlog == True):
                print("Output=", dt_now, "at", istore, time_store[istore])
            avg_state = avg_state / istep_store
            nc_out.variables['pc'][istore, :] = np.real(avg_state[:])
            istore = istore + 1
            avg_state[:] = avg_state * 0;
            istep_store = 0
            if (istore >= nstore):
                break
    nc_out.close()


def pc_to_var_TLL(fname_pc, fnames_eof, fname_out, varname_out, vartable, timename="time", modename="mode",
                  varname_pc="pc", missing_value=-9999.9, modes=""):
    # Read PC files
    nc_in = ncdf.Dataset(fname_pc, 'r')
    time = nc_in.variables[timename][:]
    timeunits = nc_in.variables[timename].units
    try:
        timecalendar = nc_in.variables[timename].calendar
    except AttributeError:
        timecalendar = 'standard'
    mode = nc_in.variables[modename][:]
    var_pc = nc_in.variables[varname_pc][:]
    if (modes == ""):
        modes = nc_in.modes
        if (np.size(modes) == 1):
            modes = np.reshape(modes, (1))
    nc_in.close()
    mode = np.asarray(mode)

    nmode = len(mode);
    ntime = len(time)
    im1 = 0;
    im2 = 0
    lonname = "lon";
    latname = "lat"
    istart = 0
    npcs = len(vartable)
    for ifile in range(0, npcs):
        im1 = vartable[ifile][0]
        im2 = vartable[ifile][1]
        # if (vartable[ifile] ==0):
        #     im1=0
        # else:
        #     im1=sum(modes[0:vartable[ifile]])
        # if (vartable[ifile] == len(modes)-1):
        #     im2=sum(modes)
        # else:
        #     im2=sum(modes[0:vartable[ifile+1]])
        data_out = var_pc[:, im1:im2]

        #        nc_eof=ncdf.Dataset(fnames_eof[vartable[ifile]],'r')
        nc_eof = ncdf.Dataset(fnames_eof[ifile], 'r')
        lon = nc_eof.variables[lonname][:]
        # lonunits = nc_eof.variables[lonname].units
        lat = nc_eof.variables[latname][:]
        # latunits = nc_eof.variables[latname].units
        eof_vec = nc_eof.variables['eof_vec'][:, :]
        mask_eof = nc_eof.variables['mask'][:, :]
        zweight_eof = nc_eof.variables['zweight'][:, :]
        nc_eof.close()
        if (istart == 0):
            nlon = len(lon);
            nlat = len(lat)
            nc_out = ncdf.Dataset(fname_out, "w")
            nc_out.createDimension(timename, None)
            nc_out.createDimension(latname, nlat)
            nc_out.createDimension(lonname, nlon)
            nc_out.createVariable(lonname, lon.dtype, (lonname))
            # nc_out.variables[lonname].units = lonunits
            nc_out.createVariable(latname, lat.dtype, (latname))
            # nc_out.variables[latname].units = latunits
            nc_out.createVariable(timename, time.dtype, (timename))
            nc_out.variables[timename].units = timeunits
            nc_out.variables[timename].calendar = timecalendar
            nc_out.createVariable(varname_out, eof_vec.dtype, (timename, latname, lonname))
            nc_out.variables[varname_out].missing_value = missing_value

            nc_out.variables[lonname][:] = lon[0:nlon]
            nc_out.variables[latname][:] = lat[0:nlat]
            nc_out.variables[timename][:] = time[0:ntime]
            var_sum = np.zeros((ntime, nlat, nlon)) * (missing_value)
            mask_sum = np.zeros((nlat, nlon))
            istart = 1

        var_tmp = np.zeros((1, nlat, nlon)) * (missing_value)
        mask_sum[mask_eof == 1.0] = 1.0
        ind = np.where(mask_eof == 1.0)
        for i in range(0, ntime):
            nend = im2 - im1
            de = eof_vec[0:nend, ind[0], ind[1]]
            var = np.dot(data_out[i:i + 1, 0:nend], de)
            var_tmp[0, ind[0], ind[1]] = var
            var_sum[i, :, :] = var_sum[i, :, :] + var_tmp[0, :, :] * zweight_eof
    var_sum = var_sum * mask_sum + (1.0 - mask_sum) * missing_value
    var_sum = np.ma.masked_invalid(var_sum)
    nc_out.variables[varname_out][:, :, :] = var_sum[:, :, :]
    nc_out.close()

