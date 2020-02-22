# This code uses the Runge Kutta method (RK4) to solve for the SHG equations
## Constants
import pdb
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt


# This code uses the Runge Kutta method (RK4) to solve for the SHG equations
## Constants
class SHG_solve:
    def __init__(self):
        self.hp = 6.626e-34
        self.c = 299792458
        self.Q_0a = 9e5
        self.Q_ca = 1e6
        init = np.linspace(-2, 2, 30)
        self.Qca_vary = self.Q_0a * (10 ** init)#np.array([9e5])  #np.array([9e5])  # self.Q_0a*(10**init) # np.linspace(1e4,1e7,30)
        self.Q_la = (self.Q_0a ** -1 + self.Q_ca ** -1) ** -1
        self.Qla_vary = (self.Qca_vary ** -1 + self.Q_0a ** -1) ** -1
        self.Q_0b = 2e5
        self.Q_cb = 1e6
        self.Qcb_vary = self.Q_0b * (10 ** init)#np.array([2e5])#  np.array([2e5])# self.Q_0b*(10**init) # np.array([3e5])# np.linspace(1e4,1e7,30)  # np.array([354482.7586206897])
        self.Q_lb = (self.Q_0b ** -1 + self.Q_cb ** -1) ** -1
        self.Qlb_vary = (self.Qcb_vary ** -1 + self.Q_0b ** -1) ** -1
        self.fa = self.c / 1550e-9
        self.fb = self.c / 775e-9
        self.ka = self.fa / (self.Q_la)
        self.ka_vary = self.fa / self.Qla_vary
        self.kb = self.fb / (self.Q_lb)
        self.kb_vary = self.fb / self.Qlb_vary
        self.fp = self.c / 1550e-9
        self.fpb = self.c / 775e-9
        self.g = 80e3
        self.P_p = 10e-3
        self.P_inb = 50e-3
        self.Pin_array = np.linspace(10e-6, 50e-3, 1000)
        self.epsilon_p = np.sqrt(
            self.ka * self.P_p / (self.hp * self.fp))  # number of photons per second injected into the cavity
        self.epsilon_pb = np.sqrt(self.kb * self.P_inb / self.hp * self.fpb)
        self.epsilonp_vary = np.sqrt(2 * (self.fp / self.Qca_vary) * self.P_p / (self.hp * self.fp))
        self.epsilonpb_vary = np.sqrt(2 * (self.fpb / self.Qcb_vary) * self.P_inb / (self.hp * self.fpb))
        self.Nphoton = 10
        self.h = (self.Nphoton / self.epsilon_p)  # units of time e-15 corresponds to injecting 1000 photons at a time
        self.hb = (self.Nphoton / self.epsilon_pb)
        self.h_vary = self.Nphoton / self.epsilonp_vary
        self.hb_vary = self.Nphoton / self.epsilonpb_vary
        self.a = 0
        self.b = 0
        self.T = self.h_vary[0] * 1e4  # 10e-9 # Number of real seconds to simulate
        self.Tb = self.hb_vary[0] * 1e6
        self.N = int(self.T // self.h_vary[0])
        self.Nb = int(self.Tb // self.hb_vary[0])
        self.track = {}
        self.track["a"] = []
        self.track["b"] = []
        self.track_a = []
        self.track_b = []
        self.track["time"] = []
        self.track["transmission"] = []
        self.track["shg"] = []
        self.track['efficiency'] = []
        self.track['sdc'] = []
        self.efficiency = np.zeros(shape=(len(self.Qca_vary), len(self.Qcb_vary)))
        # pdb.set_trace()

    def k_a(self, h_temp, a_temp, numA=0, numB=0):
        value = h_temp * ((-1j * (self.fa - self.fp) - self.ka_vary[numA]) * a_temp -
                          1j * 2 * self.g * np.conj(a_temp) * self.b - 1j * self.epsilonp_vary[numA])
        return value

    def k_b(self, h_temp, b_temp, numA=0, numB=0):
        value = h_temp * ((-1j * (self.fb - 2 * self.fp) - self.kb_vary[numB]) * b_temp -
                          1j * self.g * self.a ** 2)
        return value

    def rk4_a(self, numA, numB):
        a = self.a

        k1 = self.kb_a(h_temp=self.h_vary[numA], a_temp=a, numA=numA, numB=numB)
        k2 = self.kb_a(h_temp=self.h_vary[numA], a_temp=a + k1 / 2, numA=numA, numB=numB)
        k3 = self.kb_a(h_temp=self.h_vary[numA], a_temp=a + k2 / 2, numA=numA, numB=numB)
        k4 = self.kb_a(h_temp=self.h_vary[numA], a_temp=a + k3, numA=numA, numB=numB)

        self.a = self.a + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4_b(self, numA=0, numB=0):
        b = self.b

        k1 = self.k_b(h_temp=self.h_vary[numA], b_temp=b, numA=numA, numB=numB)
        k2 = self.k_b(h_temp=self.h_vary[numA], b_temp=b + k1 / 2, numA=numA, numB=numB)
        k3 = self.k_b(h_temp=self.h_vary[numA], b_temp=b + k2 / 2, numA=numA, numB=numB)
        k4 = self.k_b(h_temp=self.h_vary[numA], b_temp=b + k3, numA=numA, numB=numB)

        self.b = self.b + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def kb_a(self, h_temp, a_temp, numA=0, numB=0):
        value = h_temp * ((-1j * (self.fa - self.fp) - self.ka_vary[numA]) * a_temp -
                          1j * 2 * self.g * np.conj(a_temp) * self.b - 1j * np.sqrt(2 * self.fp / self.Q_0a) * 0.5 -
                          1j * np.sqrt(2 * self.fp / self.Qca_vary[numA]) * 0.5)
        return value  # included vacuum noise for initial input in mode a

    def kb_b(self, h_temp, b_temp, numA=0, numB=0):
        value = h_temp * ((-1j * (self.fb - 2 * self.fp) - self.kb_vary[numB]) * b_temp -
                          1j * self.g * self.a ** 2 - 1j * self.epsilonpb_vary[numB])
        return value

    def rk4b_a(self, numA, numB):
        a = self.a

        k1 = self.kb_a(h_temp=self.hb_vary[numA], a_temp=a, numA=numA, numB=numB)
        k2 = self.kb_a(h_temp=self.hb_vary[numA], a_temp=a + k1 / 2, numA=numA, numB=numB)
        k3 = self.kb_a(h_temp=self.hb_vary[numA], a_temp=a + k2 / 2, numA=numA, numB=numB)
        k4 = self.kb_a(h_temp=self.hb_vary[numA], a_temp=a + k3, numA=numA, numB=numB)

        self.a = self.a + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4b_b(self, numA=0, numB=0):
        b = self.b

        k1 = self.kb_b(h_temp=self.hb_vary[numA], b_temp=b, numA=numA, numB=numB)
        k2 = self.kb_b(h_temp=self.hb_vary[numA], b_temp=b + k1 / 2, numA=numA, numB=numB)
        k3 = self.kb_b(h_temp=self.hb_vary[numA], b_temp=b + k2 / 2, numA=numA, numB=numB)
        k4 = self.kb_b(h_temp=self.hb_vary[numA], b_temp=b + k3, numA=numA, numB=numB)

        self.b = self.b + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run_sim_opo(self):
        count = 0
        time = 0
        for numB, Qcb in enumerate(self.Qcb_vary):
            self.Tb = self.hb_vary[numB] * 1e5
            self.Nb = int(self.Tb // self.hb_vary[numB])
            for numA, Qca in enumerate(self.Qca_vary):
                #                 pdb.set_trace()
                for i in range(self.Nb):
                    self.rk4b_b(numA, numB)
                    self.rk4b_a(numA, numB)

                    #             pdb.set_trace()
                    if not (i % int(self.Nb // 100)):
                        self.track['a'].append(abs(self.a) ** 2)  # *self.hp*self.fa
                        self.track['b'].append(abs(self.b) ** 2)  # *self.hp*self.fb
                        transmission = self.hp * self.fb * abs(np.sqrt(self.P_inb / (self.hp * self.fb))
                                                               - 1j * np.sqrt(self.fb / self.Q_0b) * self.b) ** 2
                        self.track['transmission'].append(transmission)
                        sdc = (self.fa / self.Qca_vary[numA]) * abs(self.a) ** 2 * self.hp * self.fa
                        self.track['sdc'].append(sdc)
                        if not (i == 0):
                            time += self.hb * int(self.Nb // 100)
                            self.track['time'].append(time)
                        else:
                            self.track['time'].append(0)
                            #                 print(count)
                            #                 self.track['efficiency'].append(self.track['shg'][-1]/self.P_p)
                self.efficiency[numA][numB] = self.track['sdc'][-1] / self.P_inb
                # plt.plot(self.track['a'])
                # plt.show()
                # pdb.set_trace()
                self.track['a'].clear()
                self.track['b'].clear()
                self.track['transmission'].clear()
                self.track['shg'].clear()
                self.track['time'].clear()
                self.a = 0
                self.b = 0

            count += 1
            print(count)

    def run_sim(self):

        count = 0
        time = 0

        for numA, Qca in enumerate(self.Qca_vary):
            self.T = self.h_vary[numA] * 1e4
            self.N = int(self.T // self.h_vary[numA])
            for numB, Qcb in enumerate(self.Qcb_vary):
                #                 pdb.set_trace()
                for i in range(self.N):
                    self.rk4_a(numA, numB)
                    self.rk4_b(numA, numB)
                    pdb.set_trace()
                    if not (i % int(self.N // 100)):
                        self.track['a'].append(abs(self.a) ** 2 * self.hp * self.fa)
                        self.track['b'].append(abs(self.b) ** 2 * self.hp * self.fb)
                        transmission = self.hp * self.fa * abs(np.sqrt(self.P_p / (self.hp * self.fa))
                                                               - 1j * np.sqrt(self.fa / self.Q_0a) * self.a) ** 2
                        self.track['transmission'].append(transmission)
                        shg = (self.fb / self.Qcb_vary[numB]) * abs(self.b) ** 2 * self.hp * self.fb
                        self.track['shg'].append(shg)
                        if not (i == 0):
                            time += self.h * int(self.N // 100)
                            self.track['time'].append(time)
                        else:
                            self.track['time'].append(0)
                            #                 print(count)

                            #                 self.track['efficiency'].append(self.track['shg'][-1]/self.P_p)
                self.efficiency[numA][numB] = self.track['shg'][-1] / self.P_p
                #                 pdb.set_trace()
                self.track['a'].clear()
                self.track['b'].clear()
                self.track['transmission'].clear()
                self.track['shg'].clear()
                self.track['time'].clear()
                self.a = 0
                self.b = 0

            count += 1
            print(count)


