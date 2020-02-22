from chi2solve import *
from matplotlib import pyplot as plt
from twm import *

# a = SHG_solve()
# a.run_sim_opo()
# plt.imshow(a.efficiency,origin='lower',extent=[-2,2,-2,2],interpolation='spline36')
# plt.xlabel('Log$_{10}$($Q_{cb}/Q_{0b}$)')
# plt.ylabel('Log$_{10}$($Q_{ca}/Q_{0a}$)')
# plt.show()
# print('Qca = ' + str(a.Qca_vary[np.where(a.efficiency == np.amax(a.efficiency))[0][0]]))
# print('Qcb = ' + str(a.Qcb_vary[np.where(a.efficiency == np.amax(a.efficiency))[1][0]]))
# print('max efficiency = ' + str(2*np.amax(a.efficiency)))

# a =twm(Pin_a=10e-3,Pin_b=30e-3,slices=1000,Nphotons=10,simPts=1e5,Qc_n=10,Qcb=10e8,Qcc=10e8)
# a =twm(Pin_a=50e-3,Pin_b=30e-3,slices=1000,Nphotons=100,simPts=1e5,Qc_n=30)
a =twm(Pin_a=100e-3,Pin_b=100e-3,slices=1000,Nphotons=10,simPts=1e5,
	log_ratio=[-2,2],Qc_n=100,Q0a=6e5,Q0b=5e5,Q0c=6e5)
# a.dfg_solve()
# a.plot_efficiency()
a.steady_state_b()
# a.load_data('00.csv')
# plt.plot(a.load_data['c'])

# print(a.load_data['b'][-1])
# print(a.load_data['c'][-1])
# print(a.load_data['a'][-1])
# plt.show()
# a.plot_efficiency()
# plt.plot(a.track['Pa'])
# plt.show()
# plt.plot(a.track['Pb'])
# plt.show()
# plt.plot(a.track['Pc'])
# plt.show()
# pdb.set_trace()