# importanto bibliotecas:
import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from torch.autograd.functional import jacobian
from torch import tensor, Tensor
import sympy as sp
from sympy import *

S, X, P, S_in, mumax, ks, yxs, yps, kdec, d = sp.symbols('S, X, P, S_in, mumax, ks, yxs, yps, kdec, d', real=True)
f1 = d*(S_in-S)-(1/yxs)*(mumax*(S/(ks+S))*X)
f2 = d*(-X) + (mumax*(S/ks+S)-kdec)*X
f3 = d*(-P) + yps*(mumax*(S/(ks+S))*(1/yxs)*X)
f1s = diff(f1, S)
f1x = diff(f1, X)
f1p = diff(f1, P)
f2s = diff(f2, S)
f2x = diff(f2, X)
f2p = diff(f2, P)
f3s = diff(f3, S)
f3x = diff(f3, X)
f3p = diff(f3, P)
F = sp.Matrix([f1, f2, f3])
print(F.jacobian([S, X, P]))

def jac (t, x, params):
  S = x[0]
  X = x[1]
  P = x[2]
  
  S_in = params[0]
  mu_max_X = params[1] #dia^-1
  K_S = params[2] #g_DQO_S/m^3
  Y_X_S = params[3] #g_DQO_X/g_DQO_S
  Y_P_S = params[4] #g_DQO_P/g_DQO_S
  k_dec = params[5] #dia^-1
  D = params[6] #dia^-1
  
  dS_dS = S*X*mu_max_X/(Y_X_S*(S+K_S)**2) - X*mu_max_X/(Y_X_S*(S+K_S)) - D
  dS_dX = -S*mu_max_X/(Y_X_S*(S+K_S))
  dS_dP = 0
  dX_dS = X*mu_max_X*(1+1/K_S)
  dX_dX = -D-k_dec+mu_max_X*(S+S/K_S)
  dX_dP = 0
  dP_dS = -S*X*mu_max_X*Y_P_S/(Y_X_S*(S+K_S)**2) + X*mu_max_X*Y_P_S/(Y_X_S*(S+K_S))
  dP_dX = S*mu_max_X*Y_P_S/(Y_X_S*(S+K_S))
  dP_dP = -D
  
  jac_m = [[dS_dS, dS_dX, dS_dP],
           [dX_dS, dX_dX, dX_dP],
           [dP_dS, dP_dX, dP_dP]]
  
  return jac_m

def model (t, x, params):
    
    # S_in, mu_max_X, K_S, Y_X_S, Y_P_S, k_dec, V, q = parameters_model

    S = x[0] #g_DQO_S/m^3
    X = x[1] #g_DQO_X/m^3
    P = x[2] #g_DQO_P/m^3

    #parâmetros do modelo
    S_in = params[0]
    mu_max_X = params[1] #dia^-1
    K_S = params[2] #g_DQO_S/m^3
    Y_X_S = params[3] #g_DQO_X/g_DQO_S
    Y_P_S = params[4] #g_DQO_P/g_DQO_S
    k_dec = params[5] #dia^-1
    D = params[6] #dia^-1
   
    ##definindo a reação
    mu = mu_max_X*S/(K_S + S) #dia^-1
      
    dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
    dX_dt = (D)*(-X) + (mu-k_dec)*X
    dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X) 

    return dS_dt, dX_dt, dP_dt

##definindo os parâmetros declarados como argumento para o solve_ivp
params = (0, 1.7985e-4, 3.9974, 0.00634, 0.84113924, 0.01, 0)

#condições iniciais das variáveis dos balanços
x = [42.5000, 25.2000, 0.0]

#rotina solve_ivp

#intervalo de integração
t_step = [0., 240.]


#execução da função
DOP853 = scipy.integrate.solve_ivp(model, t_step, x, 'DOP853', args=[params], jac=jac, rtol=3e-14, atol=1e-18)
RK45 = scipy.integrate.solve_ivp(model, t_step, x, 'RK45', args=[params],jac=jac, rtol=3e-14, atol=1e-18)
Radau = scipy.integrate.solve_ivp(model, t_step, x, 'Radau', args=[params], jac=jac, rtol=3e-14, atol=1e-18)
LSODA = scipy.integrate.solve_ivp(model, t_step, x, 'RK45', args=[params], jac=jac, t_eval=np.linspace(0, 240, 25), rtol=3e-14, atol=1e-18)

#abrindo a tupla retornada em três variáveis
S_solve_ivp, X_solve_ivp, P_solve_ivp = DOP853.y

S_RK45, X_RK45, P_RK45 = RK45.y

S_radau, X_radau, P_radau = Radau.y

S_LSODA, X_LSODA, P_LSODA = LSODA.y

print("DOP853 resultado \n")
print(DOP853)
print("RK45 resultado \n")
print(RK45)
print('\n Radau resultado \n')
print(Radau)
print('LSODA resultado')
print(LSODA)



dados_P = pd.read_excel(r"C:\Users\claro\OneDrive\Documentos\Modelos Personalizados do Office\Exp Rao et al .xlsx", header=None, names=['tempo', 'concentração'], decimal=',')
dados_S = pd.read_excel(r"C:\Users\claro\OneDrive\Documentos\UTFPR\TCC\Código\Rao et al. substrato.xlsx", header=None, names=['tempo', 'concentração'], decimal=',')
dados_P['concentração'] = dados_P['concentração']/1000
dados_S['concentração'] = dados_S['concentração']/1000
t_measure = np.linspace(0, 240, 25)
#plotando 
plt.figure()
plt.scatter(t_measure, dados_P['concentração'], marker='o', color='b', label='measured data', s=75)
plt.plot(RK45.t, P_RK45)
plt.xlim([0, max(t_measure)])
plt.ylim
plt.show()
figure, ax = plt.subplots(3,3)

ax[0][0].plot(DOP853.t, S_solve_ivp, color='red')
ax[1][0].plot(DOP853.t, X_solve_ivp, color='green')
ax[2][0].plot(DOP853.t, P_solve_ivp, color='blue')
ax[0][1].plot(LSODA.t, S_LSODA, color='olive')
ax[1][1].plot(LSODA.t, X_LSODA, color='brown')
ax[2][1].plot(LSODA.t, P_LSODA, color='pink')
ax[0][2].plot(Radau.t, S_radau)
ax[1][2].plot(Radau.t, X_radau)
ax[2][2].plot(Radau.t, P_radau)
figure.tight_layout()
plt.show()

def modeltorch (t, x, params):
    
    # S_in, mu_max_X, K_S, Y_X_S, Y_P_S, k_dec, V, q = parameters_model

    S = x[0] #g_DQO_S/m^3
    X = x[1] #g_DQO_X/m^3
    P = x[2] #g_DQO_P/m^3

    #parâmetros do modelo
    S_in = params[0]
    mu_max_X = params[1] #dia^-1
    K_S = params[2] #g_DQO_S/m^3
    Y_X_S = params[3] #g_DQO_X/g_DQO_S
    Y_P_S = params[4] #g_DQO_P/g_DQO_S
    k_dec = params[5] #dia^-1
    D = params[6] #dia^-1
   
    ##definindo a reação
    mu = mu_max_X*S/(K_S + S) #dia^-1
    
    

    dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
    dX_dt = (D)*(-X) + (mu-k_dec)*X
    dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X)


    return (dS_dt, dX_dt, dP_dt)
x = tensor(x)

print(jacobian(modeltorch,(tensor(t_step))))
