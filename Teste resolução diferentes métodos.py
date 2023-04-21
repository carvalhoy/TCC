# importanto bibliotecas:
import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt

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
params = (100., 0.173489, 649186., 0.626418, 1.611689, 0.02083721, 0.0)
params2 = (1.46484375e-03, 6.11696777e+00, 2.82084961e+02,   5.57128906e-01,
  4.22119141e-01, 2.41699219e-03, 0.)
params3 = (0.37646484375, 5.2501708984375, 29.2236328125, 0.13935546875, 0.54638671875, 0.415771484375, 0.0096337890625)

#condições iniciais das variáveis dos balanços
x = [42500.0, 25200.0, 0.0]

#rotina solve_ivp

#intervalo de integração
t_step = [0, 240]

#execução da função
DOP853 = scipy.integrate.solve_ivp(model, t_step, x, 'DOP853', args=[params3], max_step=0.05)
RK45 = scipy.integrate.solve_ivp(model, t_step, x, 'RK45', args=[params3])
Radau = scipy.integrate.solve_ivp(model, t_step, x, 'Radau', args=[params3])
LSODA = scipy.integrate.solve_ivp(model, t_step, x, 'LSODA', args=[params3], t_eval=np.linspace(0, 240, 25))

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

#plotando 
figure, ax = plt.subplots(3,3)

ax[0][0].plot(DOP853.t, S_solve_ivp, 'o', color='red')
ax[1][0].plot(DOP853.t, X_solve_ivp, 'o', color='green')
ax[2][0].plot(DOP853.t, P_solve_ivp, 'o', color='blue')
ax[0][1].plot(LSODA.t, S_LSODA, 'o', color='olive')
ax[1][1].plot(LSODA.t, X_LSODA, 'o', color='brown')
ax[2][1].plot(LSODA.t, P_LSODA, 'o', color='pink')
ax[0][2].plot(Radau.t, S_radau)
ax[1][2].plot(Radau.t, X_radau)
ax[2][2].plot(Radau.t, P_radau)
plt.show()
