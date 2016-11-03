import numpy as np
import matplotlib.pyplot as plt
import functions as fun
import sys


ev=27.2114e0
amu=0.00054858e0

input_file=open(str(sys.argv[1]),'r')

print('<< X-ray python tools >>\n')
print('input file: ',str(sys.argv[1]),'\n')

x=None
eloss=False

for line in input_file:
    words = line.split()
    if(words[0] == 'calc'):
        runtype=words[1]
    elif(words[0] == 'mass' or words[0] == 'm' or words[0] == 'mu'):
        mu=float(words[1])
    elif(words[0] == 'omega_in' or words[0] == 'omega'):
        omega_in=float(words[1])
    elif(words[0] == 'Gamma' or words[0] == 'gamma'):
        Gamma=float(words[1])
    elif(words[0] == 'plot'):
        plot_graph=True
    elif(words[0] == 'eloss' or words[0] == 'Eloss'):
        eloss=True
    elif(words[0] == 'dump'):
        dump_data=True
    elif(words[0] == 'xrange'):
        xi=float(words[1])
        xf=float(words[1])
        nx=int(words[1])
        x=np.linspace(xi,xf,nx,dtype=float)
    elif(words[0] == 'init_pot'):
        if(words[1]=='harm'):
            pot_g='harm'
            freq_g=float(words[2])
            x0_g=float(words[3])
            V_g_min=float(words[4])
        else:
            pot_g='harm'
            x,V_g=np.genfromtxt(words[2],dtype=float,skipheader=1,unpack=True)
    elif(words[0] == 'decaying_pot'):
        if(words[1]=='harm'):
            pot_c='harm'
            freq_c=float(words[2])
            x0_c=float(words[3])
            V_c_min=float(words[4])
        else:
            pot_c=words[2]
            x,V_c=np.genfromtxt(words[2],dtype=float,skipheader=1,unpack=True)
    elif(words[0] == 'final_pot'):
        if(words[1]=='harm'):
            pot_f='harm'
            freq_f=float(words[2])
            x0_f=float(words[3])
            V_f_min=float(words[4])
        else:
            pot_f=words[2]
            x,V_f=np.genfromtxt(words[2],dtype=float,skipheader=1,unpack=True)
        

if x is None:
    x = np.linspace(x0_g-5.0,x0_g+5.0,512,dtype=float)

#---------------------------------------------------------------------------
print('input values:')
print("runtype: ",runtype)
print("mass: ",mu,"amu")
print("Gamma: ",Gamma,"eV")

print("initial potential: ",pot_g)
if(pot_g=='harm'):
    print("   ",freq_g," eV ",x0_g," a.u. ",V_g_min," eV ")
    V_g=fun.harmonic_pot(freq_g/ev,mu/amu,x,x0_g) + V_g_min/ev

print("initial potential: ",pot_c)
if(pot_c=='harm'):
    print("   ",freq_c," eV ",x0_c," a.u. ",V_c_min," eV ")
    V_c=fun.harmonic_pot(freq_c/ev,mu/amu,x,x0_c) + V_c_min/ev

if(runtype=='rixs'):
    print("initial potential: ",pot_f)
    if(pot_f=='harm'):
        print("   ",freq_f," eV ",x0_f," a.u. ",V_f_min," eV ")
        V_f=fun.harmonic_pot(freq_f/ev,mu/amu,x,x0_f) + V_f_min/ev

    print("incoming photon frequency: ",omega_in," eV")
#---------------------------------------------------------------------------

if(runtype=='xas'):
    omega,sig_xas=fun.compute_xas(mu/amu,x,V_g,V_c,Gamma/ev)
    if(plot_graph):
        plt.plot(omega*ev,sig_xas,'-')
        plt.show()
elif(runtype=='rixs'):
    omega_out,sig_rixs=fun.compute_rixs(omega_in/ev,mu/amu,x,V_g,V_c,V_g,Gamma/ev)
    if(plot_graph and eloss):
        plt.plot(omega_in - (omega_out*ev),sig_rixs,'-')
        plt.show()
    else:
        plt.plot((omega_out)*ev,sig_rixs,'-')
        plt.show()
