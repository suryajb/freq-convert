from chi2solve import *

class twm(SHG_solve):
    def __init__(self, lambda_b=777e-9, lambda_c=None, lambda_a=1558e-9, log_ratio=None, Qc_n=10,
                 lambda_fa=np.array([1558e-9]), lambda_fb=np.array([777e-9]), lambda_fc=None, Q0a=1e6, Q0b=1e6, Q0c=1e6,
                 Qca=1e6, Qcb=1e6, Qcc=1e6, Pin_a=0.0, Pin_b=0.0, Pin_c=0.0, Nphotons=100, simPts=1e4, slices=1000):
        # log_ratio should be a list with 2 elements
        # lambda_fa default is definied to be 1 / (1 / lambda_b - 1 / lambda_c)
        # Pin_a/b/c can be inputted as float or array

        self.a = 0
        self.b = 0
        self.c = 0
        self.C = 299792458  # speed of light
        # some constants
        self.lambda_a = lambda_a
        self.hbar = 1.054571817e-34
        self.g = 2*np.pi*80e3
        if log_ratio == None:
            self.init_n = np.linspace(-2, 2, Qc_n)
            self.log_ratio=log_ratio
        else:
            assert (len(log_ratio) == 2)
            self.init_n = np.linspace(log_ratio[0], log_ratio[1], Qc_n)
            self.log_ratio=log_ratio
        self.lambda_fa = lambda_fa
        self.lambda_fb = lambda_fb
        if lambda_c == None:  # Defined None before, because this ensures that lambda_c is in resonance
            self.lambda_c = (1 / lambda_b - 1 / lambda_a) ** -1
        if lambda_fc == None:
            self.lambda_fc = (1 / lambda_fb - 1 / lambda_fa) ** -1
        self.slices = slices

        # parameters for mode a
        self.w_a = 2 * np.pi * self.C / self.lambda_a  # mode a frequency
        self.w_fa = 2 * np.pi * self.C / self.lambda_fa  # pump a frequency
        self.Q0a = Q0a
        if log_ratio:
            self.Qca = self.Q0a * (10 ** self.init_n)
        else:
            self.Qca = np.array([Qca])
        self.Qla = (self.Q0a ** -1 + self.Qca ** -1) ** -1
        if len(self.Qla) > 1 and len(self.w_fa) > 1:
            raise ValueError('should only sweep one parameter, Qc or wavelength')
        self.ka = self.w_a / self.Qla  # array for total dissipation rate
        self.k0a = self.w_a / self.Q0a  # float for rate of internal loss
        self.k1a = self.w_a / self.Qca  # array for rate of external coupling

        # parameters for mode b
        self.lambda_b = lambda_b
        self.w_b = 2 * np.pi * self.C / self.lambda_b
        self.w_fb = 2 * np.pi * self.C / self.lambda_fb  # pump b frequency
        self.Q0b = Q0b
        if log_ratio:
            self.Qcb = self.Q0b * (10 ** self.init_n)
        else:
            self.Qcb = np.array([Qcb])
        self.Qlb = (self.Q0b ** -1 + self.Qcb ** -1) ** -1
        if len(self.Qlb) > 1 and len(self.w_fb) > 1:
            raise ValueError('should only sweep one parameter, Qc or wavelength')
        self.kb = self.w_b / self.Qlb  # array
        self.k0b = self.w_b / self.Q0b
        self.k1b = self.w_b / self.Qcb

        # parameters for mode c
        self.w_c = self.w_b-self.w_a# 2 * np.pi * self.C / self.lambda_c
        self.w_fc = 2 * np.pi * self.C / self.lambda_fc
        self.Q0c = Q0c
        if log_ratio:
            self.Qcc = self.Q0c * (10 ** self.init_n)
        else:
            self.Qcc = np.array([Qcc])
        self.Qlc = (
                   self.Q0c ** -1 + self.Qcc ** -1) ** -1  # Qlc is an array, Q0c is a float, w_fa/b/c is an array, w_a/b/c is a float.
        if len(self.Qlc) > 1 and len(self.w_fc) > 1:
            raise ValueError('should only sweep one parameter, Qc or wavelength')
        self.kc = self.w_c / self.Qlc  # array
        self.k0c = self.w_c / self.Q0c
        self.k1c = self.w_c / self.Qcc

        # Input power parameters
        if Pin_a == 0 and Pin_b == 0 and Pin_c == 0: raise ValueError('No imput power defined')
        self.Pin_a = np.array([Pin_a]) if type(Pin_a) == float else Pin_a
        self.Pin_b = np.array([Pin_b]) if type(Pin_b) == float else Pin_b
        self.Pin_c = np.array([Pin_c]) if type(Pin_c) == float else Pin_c
        self.epsilon_pa = np.sqrt(2 * (self.k1a) * self.Pin_a / (self.hbar * self.w_fa))
        self.epsilon_pb = np.sqrt(2 * (self.k1b) * self.Pin_b / (self.hbar * self.w_fb))
        self.epsilon_pc = np.sqrt(2 * (self.k1c) * self.Pin_c / (self.hbar * self.w_fc))
        self.ha = np.sqrt(Nphotons) / self.epsilon_pa if self.epsilon_pa.any() else 1e-12 # seconds corresponding to Nphotons injected (this corresponds to a time-step)
        self.hb = np.sqrt(Nphotons) / self.epsilon_pb if self.epsilon_pb.any() else 1e-12
        self.hc = np.sqrt(Nphotons) / self.epsilon_pc if self.epsilon_pc.any() else 1e-12
        self.Ta = self.ha*simPts # number of seconds in total for simulation
        self.Tb = self.hb*simPts
        self.Tc = self.hc*simPts
        self.Na = self.Nb = self.Nc = simPts
        self.track = {}
        self.track['a'] = []
        self.track['b'] = []
        self.track['c'] = []
        self.track['time'] = []
        self.track['aTrans'] = []
        self.track['bTrans'] = []
        self.track['cTrans'] = []
        self.track['aEta'] = []
        self.track['bEta'] = []
        self.track['cEta'] = []
        self.efficiency = np.zeros(shape=(len(self.Qca), len(self.Qcb)))
        if not(type(self.Qca)==float):
            assert(len(self.Qca)==len(self.Qcc))

    def k(self,h,w,w_p,k_vary,xtemp,ytemp,ztemp,eps): # x/y/xtemp are the mode amplitudes
        val = h * ((-1j * (w - w_p) - k_vary) * xtemp - 1j * self.g * ytemp * ztemp + eps)
        return val

    def rk4(self,h,w,w_p,k_vary,val,ytemp,ztemp,eps):
        k1 = self.k(h, w, w_p, k_vary, val, ytemp, ztemp, eps)
        k2 = self.k(h, w, w_p, k_vary, val + k1 / 2, ytemp, ztemp, eps)
        k3 = self.k(h, w, w_p, k_vary, val + k2 / 2, ytemp, ztemp, eps)
        k4 = self.k(h, w, w_p, k_vary, val + k3, ytemp, ztemp, eps)
        val += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return val

    def dfg_solve(self): # this is specifically for the case where b is the input mode and c is the output mode.
        h = self.ha[-1]
        start = time.time()
        count = 0
        for i,Qcb in enumerate(self.Qcb):
            for j,Qcc in enumerate(self.Qcc):
                for k,w_fa in enumerate(self.w_fa):
                    for l,w_fb in enumerate(self.w_fb):
                        for m,w_fc in enumerate(self.w_fc):
                            for n in range(int(self.Na)):
                                self.a = self.rk4(h,self.w_a,w_fa,self.ka[j],self.a,self.b,np.conj(self.c),self.epsilon_pa[j]) # j is used for k1a because coupling Q for both modes a and c should be the same
                                self.b = self.rk4(h, self.w_b, w_fb, self.kb[i], self.b, self.a, self.c,
                                                  self.epsilon_pb[i])
                                self.c = self.rk4(h, self.w_c, w_fb - w_fa, self.kc[j], self.c, np.conj(self.a),
                                                  self.b, 0)
                                # pdb.set_trace()
                                if not (n % int(self.Na // self.slices)):
                                    self.track['a'].append(np.abs(self.a) ** 2)
                                    self.track['b'].append(np.abs(self.b) ** 2)
                                    self.track['c'].append(np.abs(self.c) ** 2)
                            self.track['a']=np.asarray(self.track['a'])
                            self.track['Pa'] = self.track['a'] * np.asarray(
                                (self.w_a / self.Qca[j]) * self.hbar * self.w_a)
                            self.track['Pb'] = self.track['b'] * np.asarray(
                                (self.w_b / self.Qcb[i]) * self.hbar * self.w_b)
                            self.track['Pc'] = self.track['c'] * np.asarray(
                                (self.w_c / self.Qcc[j]) * self.hbar * self.w_c)
                            self.efficiency[i][j] = self.track['Pc'][-1]/self.Pin_b[0]

                # pdb.set_trace()
                self.save_data(self.track, i, j)
                self.track['a'] = []; self.track['b'] = []; self.track['c'] = []
                self.track['Pa'] = []; self.track['Pb'] = []; self.track['Pc'] = []
                self.a = 0; self.b = 0; self.c = 0
                # pdb.set_trace()
                count += 1
                print(count)
        end = time.time()
        print('time elapsed = '+str(end-start)+' seconds')

    def plot_efficiency(self,sx=8,sy=8,suffix='0'):
        print('max efficiency = ' + str(2 * np.amax(self.efficiency)))
        print('Qcb = ' + str(self.Qcb[np.where(self.efficiency == np.amax(self.efficiency))[0][0]]))
        print('Qcc = ' + str(self.Qcc[np.where(self.efficiency == np.amax(self.efficiency))[1][0]]))
        print('Qlb = ' + str(self.Qlb[np.where(self.efficiency == np.amax(self.efficiency))[0][0]]))
        print('Qlc = ' + str(self.Qlc[np.where(self.efficiency == np.amax(self.efficiency))[1][0]]))
        fig, ax = plt.subplots(1, 1, figsize=(sx, sy))
        ax.set_xlabel('Log$_{10}$($Q_{cb}/Q_{0b}$)')
        ax.set_ylabel('Log$_{10}$($Q_{ca}/Q_{0a}$)')
        ax.imshow(self.efficiency, origin='lower', extent=[self.log_ratio[0], self.log_ratio[1], self.log_ratio[0], self.log_ratio[1]], interpolation='spline36')
        plt.show()
        ax.imshow(self.efficiency, origin='lower', extent=[self.log_ratio[0], self.log_ratio[1], self.log_ratio[0], self.log_ratio[1]], interpolation='spline36')
        fig.savefig('Pin_a='+str(self.Pin_a[0])+'_'+'Pin_b='+str(self.Pin_b[0])+'efficiency'+suffix+'.png', bbox_inches="tight")

    def wavelength_sweep(self,lambda_start=None,lambda_end=None,N=None,marker=None):
        if lambda_start == None or lambda_end == None or N == None or marker == None:
            raise ValueError('specify start and end lambda, # of points, and which mode (a,b,c)')
        elif marker=='a':
            original_fa = self.w_fa.copy()
            self.w_fa = np.linspace(lambda_start,lambda_end,N)
            self.dfg_solve()
            self.w_fa = original_fa.copy()
        elif marker == 'b':
            original_fb = self.w_fb.copy()
            self.w_fb = np.linspace(lambda_start, lambda_end, N)
            self.dfg_solve()
            self.w_fb = original_fb.copy()
        elif marker == 'c':
            original_fc = self.w_fc.copy()
            self.w_fc = np.linspace(lambda_start, lambda_end, N)
            self.dfg_solve()
            self.w_fc = original_fc.copy()
        else:
            print('unknown marker')

    def steady_state_b(self):
        # solves for the steady state solution.
        self.ss_b,self.ss_c,self.ss_a = np.zeros(shape=(len(self.Qca), len(self.Qcb))),np.zeros(shape=(len(self.Qca), len(self.Qcb))),np.zeros(shape=(len(self.Qca), len(self.Qcb)))
        for i,Qcb in enumerate(self.Qcb):
            for j,Qcc in enumerate(self.Qcc): # for ka/kc use j as an index
                A = self.ka[j]**2 * self.kc[j]**2
                B = 2*self.g**2 * self.ka[j] * self.kc[j]
                C = self.g**4
                D = self.g**2 * np.abs(self.epsilon_pa[j])**2 * self.kc[j]
                zeroth = np.abs(self.epsilon_pb[i])**2 * A**2
                first = -2 * np.abs(self.epsilon_pb[i])**2 * A * B - (D**2 + 2 * self.kb[i]*D*A + self.kb[i]**2 * A**2)
                second = np.abs(self.epsilon_pb[i])**2*(2*A*C + B**2) + (2*self.kb[i]*D*B + 2*self.kb[i]**2 * A * B)
                third = -2*np.abs(self.epsilon_pb[i])**2 * B * C - (self.kb[i]**2 * (2*A*C + B**2) + 2*self.kb[i] * D * C)
                fourth = np.abs(self.epsilon_pb[i])**2 * C**2 + 2*self.kb[i]**2 * B * C
                fifth = -1*self.kb[i]**2 * C**2
                coeffs = [fifth,fourth,third,second,first,zeroth]
                self.roots = np.roots(coeffs)
                print(np.real(self.roots[np.isreal(self.roots)])[np.where(np.real(self.roots[np.isreal(self.roots)])>0)])
                self.ss_b[i][j] = np.amin(np.real(self.roots[np.isreal(self.roots)])[np.where(np.real(self.roots[np.isreal(self.roots)])>0)])
                self.ss_c[i][j] = self.ss_b[i][j] * self.g**2 * np.abs(self.epsilon_pa[j])**2 / (A - B*self.ss_b[i][j] + C*self.ss_b[i][j]**2)
                self.ss_a[i][j] = np.abs(self.epsilon_pa[j])**2 * self.kc[j]**2 / (A - B*self.ss_b[i][j] + C*self.ss_b[i][j]**2)
                print(self.ss_b[i][j],self.ss_c[i][j],self.ss_a[i][j],i,j)
                self.efficiency[i][j] = self.ss_c[i][j] * np.asarray(
                    (self.w_c / self.Qcc[j]) * self.hbar * self.w_c)/self.Pin_b[0]
        if self.log_ratio:
            self.plot_efficiency()

    def save_data(self,track,i,j):
        df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in track.items()]))
        df.to_csv(str(i)+str(j)+'.csv')

    def load_data(self,filename):
        self.df = pd.read_csv(filename)
        self.load_data = self.df.to_dict('list')
