############################## 
#####author:LongGang Pang#####
##############################



from __future__ import absolute_import, division, print_function




class CLIdeal(object):
    '''The pyopencl version for 3+1D ideal hydro dynamic simulation'''
    def __init__(self,configs,handcrafted_eos=None,gpu_id=0):
    '''Params:
    :param configs: hydrodynamic configurations,from configs import cfg
    :param gpu_id:  use which gpu for the calculation if there are many per node
    '''
    #create opencl environment 
    self.cfg = configs
    self.cwd,cwf = os.path.split(__file__)  #split dictory and file
    #create the fPathOut directory if not exists
    path = self.cfg.fPathOut
    if not os.path.exists(path):
        os.makedirs(path)
    
    #choose proper real,real4,real8 sizes
    self.determine_float_size(self,cfg)

    from backend_opencl import OpenCLBackend
    self.backend = OpenCLBackend(self.cfg,gpu_id)
    
    self.ctx = self.backend.ctx
    self.queue = self.backend.default_queue

    self.size = self.cfg.NX*self.cfg.NY*self.cfg.NZ
    self.tau = self.cfg.real(self.cfg.TAU0)

    self.compile_options = self.__compile_options()

    #set eos, create eos table for interpolation
    #self.eos_table must be before __loadAndBuildCLPrg() to pass
    #table information to definitions
    if handcrafted_eos is None:
        self.eos = Eos(self.cfg.IEOS)
    else:
        self.eos = handcrafted_eos

    #the default muB on hypersf is 0,unless IEOS=1, 'PCE165'
    chemical_potential_on_hypersf(self.cfg.TFRZ,path,
                                  eos_type='ZeroChemicalPotential')

    if handcrafted_eos is not None:
        self.eos_table = self.eos.create_table(self.ctx,
                self.compile_options)
    elif self.cfg.IEOS ==1:
        self.eos_table = self.eos.create_table(self.ctx,
                self.compile_options,nrow=100,ncol=1555)
        chemical_potential_on_hypersf(self.cfg.TFRZ,path,
                                      eos_type='PCE165')                               

    elif self.cfg.IEOS ==4:
        self.eod_table = self.eos.create_table(self.ctx,
                self.compile_options, nrow=4, ncol=1001)
    else:
        self.eos_table = self.eos.create_table(self.ctx,
                self.compile_options)

    self.efrz = self.eos.f_ed(self.cfg.TFRZ)

    #store 1D and 2D bulk info at each time step
    if self.cfg.save_to_hdf5:
        from bulkinfo_h5 import BulkInfo
    else:
        from bulkinfo import BulkInfo

    self.bulkinfo = BulkInfo(self.cfg,self.ctx,self.queue,
            self.eos_table,self.compile_options)

    self.__loadAndBuildCLPrg()

    #define buffer on device side, d_ev1 stores ed,vx,vy,vz
    mf = cl.mem_flags
    self.h_ev1 = np.zeros((self.size,4), self.cfg.real)

    self.d_ev = [cl.Buffer(self.ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=self.h_ev1),
                 cl.Buffer(self.ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=self.h_ev1),
                 cl.Buffer(self.ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=self.h_ev1),
                 cl.Buffer(self.ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=self.h_ev1) ]
    self.d_Src = cl.Buffer(self.ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=self.h_ev1)

    self.submax = np.empty(64,self.cfg.real)
    self.d_submax = cl.Buffer(self.ctx, cl.men_flags.READ_WRITE, size=self.h_ev1.nbytes)

    #d_ev_old: for hypersf calculation;
    self.d_ev_old = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)

    #d_hypersf: store the dSigma^{mu}, vx, vy, veta, tau, x, y, eta
    #on freeze out hyper surface
    self.d_hypersf = cl.Buffer(self.ctx, mf.READ_WRITE, size=1500000*self.cfg,sz_real8)

    #the position of the hyper surface in cartersian coordinates
    self.d_sf_txyz = cl.Buffer(self.ctx, mf.READ_WRITE, size=1500000*self.cfg.sz_real4)
    h_num_of_sf = np.zeros(1, np.int32)
    self.d_num_of_sf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_num_of_sf);
    
    self.history = []


    
    def determine_float_size(self, cfg):
        cfg.sz_int = np.dtype('int32').itemsize     #size of (int) in c
        if cfg.use_float32 == True :
            cfg.real = np.float32
            cfg.real4 = array.vec.float4
            cfg.real8 = array.vec.float8
            cfg.sz_real = np.dtype('float32').itemsize    #size of (float) in c 
            cfg.sz_real4 = array.vec.float4.itemsize
            cfg.sz_real8 = array.vec.float8.itemsize
        else :
            cfg.real = np.float64
            cfg.real4 = array.vec.double4
            cfg.real8 = array.vec.double8
            cfg.sz_real = np.dtype('float64').itemsize    #size of (double) in c 
            cfg.sz_real4 = array.vec.double4.itemsize
            cfg.sz_real8 = array.vec.double8.itemsize

    def load_ini(self, dat):
        '''load initial condition stored in np array whose 4 columns
           are (Ed, vx, vy, vz) and num_of_rows = NX*NY*NZ'''
        print('start to load ini data')
        self.h_ev1 = dat.astype(self.cfg.real)
        cl.enqueue_copy(self.queue, self.d_ev[1],self.h_ev1).wait()
        print('end of loading ini data')
  

    def __compile_options(self):
    optlist = ['TAU0', 'DT', 'DX', 'DY','DZ', 'ETAOS_XMIN', 'ETAOS_YMIN',\
                'ETAOS_LEFT_SLOP', 'ETAOE_RIGHT_SLOP', 'LAM1']
    gpu_defines = [ '-D %s=%sf'%(key,value) for (key,value)
            in list(self.cfg.__dict__.items()) if key in optlist ]
    gpu_defines.append('-D {key}={value}'.format(key='NX',value=self.cfg.NX))
    gpu_defines.append('-D {key}={value}'.format(key='NY',value=self.cfg.NY))
    gpu_defines.append('-D {key}={value}'.format(key='NZ',value=self.cfg.NZ))
    gpu_defines.append('-D {key}={value}'.format(key='SIZE',
                       value=self.cfg.NX*self.cfg.NY*self.cfg.NZ))

    #local memory size along x,y,z direction with 4 boundary cells
    gpu_defines.append('-D {key}={value}'.format(key='BSZ',value=self.cfg.BSZ))

    #determine float32 or double data type in *.cl file
    if self.cfg.use_float32:
        gpu_defines.append('-D USE_SINGLE_PRECISION')


    #choose EOS by ifdef in *.cl file
    if self.cfg.riemann_test:
        gpu_defines.append('-D RIEMANN_TEST')

    if self.cfg.IEOS == 4:
        '''delta ed is not constant;delta T is constant;
        using binary search for energy density'''
        gpu_defines.append( '-D EOS_BINARY_SEARCH' )  #WB2014
    else:
        gpu_defines.append( '-D EOS_TABLE' )   #WB2014

    

    #set the include path for the header file
    gpu_defines.append('-I ' +os.path.join(self.cwd, 'kernel/'))
    return gpu_defines
 

    def __loadAndBuildCLPrg(self):
    #load and build *.cl programs with compile self.compile_options
    with open(oc.path.join(self.cwd, 'kernel', 'kernel_ideal.cl'), 'r') as f:
        prg_src = f.read()
        self.kernel_ideal = cl.Program(self.ctx, prg_src).build(
                            options=' '.join(self.compile_options))


    hypersf_defines = list(self.compile_options)
    hypersf_defines.append('-D {key}={value}'.format(key='nxskip', value=self.cfg.nxskip))
    hypersf_defines.append('-D {key}={value}'.format(key='nyskip', value=self.cfg.nyskip))
    hypersf_defines.append('-D {key}={value}'.format(key='nzskip', value=self.cfg.nzskip))
    with open(os.path.join(self.cwd,'kernel', 'kernel_hypersf.cl'), 'r') as f:
        src_hypersf = f.read()
        self.kernel_hypersf = cl.Program(self.ctx, src_hypersf).build(
                                         options=' '.join(hypersf_defines))





    def stepUpdate(self, step, jet_eloss_src={'switch_on':False, 'start_pos_index':0, 'direction':0}):
        '''Do step update in kernel with KT algorithm
           Args:
                gpu_ev_old: self.d_ev[1] for the 1st step,
                            self.d_ev[2] for the 2nd step
                step: the 1st or the 2nd step in runge-kutta
        '''
    # update d_Src by KT time splitting, along=1,2,3 for 'x','y','z'
    # input: gpu_ev_old, tau, size, along_axis
    # output: self.d_Src
    NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ
    self.kernel_ideal.kt_src_christoffel(self.queue, (NX*NY*NZ, ), None,
                     self.d_Src, self.d_ev[step],self.eos_table,
                     self.tau, np.int32(step)
                     ).wait()

    self.kernel_ideal.kt_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
            self.d_Src, self.d_ev[step], self.eos_table,
            self.tau).wait()

    self.kernel_ideal.kt_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
            self.d_Src, self.d_ev[step], self.eos_table,
            self.tau).wait()
    
    self.kernel_ideal.kt_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
            self.d_Src, self.d_ev[step], self.eos_table,
            self.tau).wait()

    # if step=1, T0m' = T0m + d_Src*dt, update d_ev[2]
    # if stwp=2, T0m = T0m +0.5*dt*d_Src, update d_ev[1]
    # Notice that d_Src = f(t,x) at step1 and 
    # d_Src = ( f(t,x) + f(t+dt,x(t+dt)) ) at step2
    # output: d_ev[] where need_update=2 for step1 and 1 for step2
    self.kernel_ideal.update_ev(self.queue, (NX*NY*NZ, ), None,
                                self.d_ev[3-step], self.d_ev[1], self.d_Src,
                                self.eos_table, self.tau, np.int32(step)).wait()


    def max_energy_density(self):
    '''Calculate the maximum energy density on GPU and output the value '''
    self.kernel_reduction.reduction_stage1(self.queue, (256*64, ),(256,),
            self.d_ev[1], self.d_submax, self.d_submax).wait()
    cl.enqueue_copy(self.queue, self.submax, self.d_submax).wait()
    return self.submax.max() 

#___________________________________________________________________________
#---------------------------------------------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def ev_to_host(self):
    '''copy energy density and fluid velocity from device to host'''
    cl.enqueue_copy(self.queue, self.h_ev1, self.d_ev[1]).wait()


    def get_hypersf(self, n, ntskip):
        '''get the freeze out hyper surface from d_ev_old and d_ev_new
        global_size=(NX/nxskip, NY/nyskip, NZ/nzskip) '''
        is_finished = self.edmax < self.efrz

        if n == 0:
            cl.enqueue_copy(self.queue, self.d_ev_old,
                            self.d_ev[1]).wait()
            self.tau_old = self.cfg.TAU0
        elif (n % ntskip == 0) or is_finished:
            nx = (self.cfg.NX-1)//self.cfg.nxskip + 1
            ny = (self.cfg.NY-1)//self.cfg.nyskip + 1 
            nz = (selg.cfg.NZ-1)//self.cfg.nzskip + 1
            tau_new = self.tau
            #get dSigma, vx, vy, veta, etas on hypersf
            self.kernel_hypersf.get_hypersf(self.queue, (nx, ny, nz), None,
                    self.d_hypersf, self.d_sf_txyz, self.d_num_of_sf,
                    self.d_ev_old, self.d_ev[1],
                    self.cfg.real(self.tau_old), self.cfg.real(tau_new)).wait()

            #update with current tau and d_ev[1]
            cl.enqueue_copy(self.queue, self.d_ev_old, self.d_ev[1]).wait()
            self.tau_old = tau_new

        return is_finished

    def save(self

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#---------------------------------------------------------------------------
#___________________________________________________________________________





def main():
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    #os.environ[ 'PYOPENCL_COMPILER_OUTPUT' ] = '1'  


    from config import cfg, write_config
    print('start...')
    cfg.IEOS = 1
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 105
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.DZ = 0.16
    cfg.DT = 0.02
    cfg.ntskip = 16
    cfg.nxskip = 2
    cfg.nyskip = 2
    cfg.nzskip = 2
    cfg.Eta_gw = 0.4
    cfg.ImpctParameter = 2.4
    cfg.ETAOS = 0.0
    cfg.TFRZ = 0.137

    cfg.Edmax = 55
    cfg.TAU0 = 0.4

    cfg.fPathOut = 'ideal/results/test1/'

    cfg.save_to_hdf5 = False

    cfg.BSZ = 64

    write_comfig(cfg)

    ideal = CLIdeal(cfg,gpu_id=1)

