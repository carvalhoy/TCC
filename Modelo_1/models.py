import lmfit as lm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.integrate
import tools

### Gouveia 1
def model(t, x, params:lm.Parameters):
    
     SA = x[0] #kg_DQO_S/m^3
     SB = x[1] #kg_DQO_X/m^3
     SC = x[2] #kg_DQO_P/m^3
        
     try:
          ##parâmetros do modelo
          S_max = params['S_max'].value #L
          Kl = params['Kl'].value # dia^-1
          Kr = params['Kr'].value # dia^-1
          K2 = params['K2'].value # dia^-1
          K3 = params['K3'].value # dia^-1
          alpha = params['alpha'].value # [-]
        
    
     except KeyError:
         S_max, Kl, Kr, K2, K3, alpha = params
    
     # Operação em batelada
     # Definindo balanço de componentes:
     dSA_dt = - (alpha * Kr * SA + (1 - alpha) * Kl * SA)
     dSB_dt = (1 - alpha) * ((Kl * SA) - (K2 * SB))
     dSC_dt = (K2 * SB * (1 - alpha)) + (alpha * Kr * SA) - (K3 * SC)

     return [dSA_dt, dSB_dt, dSC_dt]

def wrapper_model(t, x, params, tsolve, Rsquare, data):
     
     try:
          ##parâmetros do modelo
          S_max = params['S_max'].value #L
          Kl = params['Kl'].value # dia^-1
          Kr = params['Kr'].value # dia^-1
          K2 = params['K2'].value # dia^-1
          K3 = params['K3'].value # dia^-1
          alpha = params['alpha'].value # [-]
    
     except KeyError:
         S_max, Kl, Kr, K2, K3, alpha = params
         
     ## Resolução do sistema de EDOs:
     sim = scipy.integrate.solve_ivp(model, t, x, 'Radau', args=[params], t_eval=tsolve)
     sA, sB, sC = sim.y
     
     ## Balanço para obter produto: 
     sD = np.zeros(sim.t.shape[0])
     for i in range(sim.t.shape[0]):
          sD[i] = S_max - sA[i] - sB[i] - sC[i]
     
     ## Calculo do R^2:
     if (Rsquare == True):
          R2_S = tools.r2(np.array(data[0]['concentração']), sA)
          R2_P = tools.r2(np.array(data[1]['concentração']), sD)
          return [R2_S, R2_P]
     
     else:
          sol = np.array([sA, sB, sC, sD, sim.t])
          return sol

### Gouveia 2
       
def modelo_analitico(t, params, Rsquare, data):
   def SA(t, S_0, alpha, kr, kl):
      Sa = S_0*(alpha*np.exp(-kr*t) + (1-alpha)*np.exp(-kl*t))
      return Sa

   def SB(t, S_0, alpha, kl, k2):
      Sb = S_0*((1-alpha)*kl*(np.exp(-kl*t) - np.exp(-k2*t))/(k2 - kl))
      return Sb

   def SC(t, S_0, alpha, kl, kr, k2, k3):
      Sc = S_0 * (alpha*kr*(np.exp(-kr*t) - np.exp(-k3*t))/(k3 - kr) + (1-alpha)* kl * k2 * ((k3 - k2)*np.exp(-kl*t) - (k3 - kl)*np.exp(-k2*t) + (k2 - kl)*np.exp(-k3*t))/((k2 - kl)*(k3 - kl)*(k3-k2)))
      return Sc    

   def SD(t, S_max, kl, kr, k2, k3, alpha):
      Sd = S_max*(alpha*(1-np.exp(-kr*t) - kr*(np.exp(-kr*t) - np.exp(-k3*t))/(k3 - kr)) + (1-alpha)*(1-np.exp(-kl*t) - kl*(np.exp(-kl*t) - np.exp(-k2*t))/(k2 - kl) - kl * k2 * ((k3 - k2)*np.exp(-kl*t) - (k3 - kl)*np.exp(-k2*t) + (k2 - kl)*np.exp(-k3*t))/((k2 - kl)*(k3 - kl)*(k3-k2))))
      return Sd
   
   try:
          ##parâmetros do modelo
          S_max = params['S_max'].value #L
          Kl = params['Kl'].value # dia^-1
          Kr = params['Kr'].value # dia^-1
          K2 = params['K2'].value # dia^-1
          K3 = params['K3'].value # dia^-1
          alpha = params['alpha'].value # [-]
    
   except KeyError:
         S_max, Kl, Kr, K2, K3, alpha = params
   
   sim_sa = SA(t, S_max, alpha, Kr, Kl)
   sim_sb = SB(t, S_max, alpha, Kl, K2)
   sim_sc = SC(t, S_max, alpha, Kl, Kr, K2, K3)
   sim_sd = SD(t, S_max, Kl, Kr, K2, K3, alpha)
   
   if (Rsquare == True):
      if np.shape(data)[0] == 3:
         R2_S = tools.r2(np.array(data[0]['concentração']), sim_sa)
         R2_I = tools.r2(np.array(data[1]['concentração']), sim_sc)
         R2_P = tools.r2(np.array(data[2]['concentração']), sim_sd)
         return [R2_S, R2_I, R2_P]
      else:
         R2_S = tools.r2(np.array(data[0]['concentração']), sim_sa)
         R2_P = tools.r2(np.array(data[1]['concentração']), sim_sd)
         return [R2_S, R2_P]
   else:      
      return [sim_sa, sim_sb, sim_sc, sim_sd]

### Gouveia 3

def model_3(t, x, params):
   
   try:
         ##parâmetros do modelo
         sAr_0 = params['sAr_0'].value #L
         sAl_0 = params['sAl_0'].value # dia^-1
         kl = params['kl'].value # dia^-1
         kr = params['kr'].value # dia^-1
         k2 = params['k2'].value # dia^-1
         k3 = params['k3'].value # [-]
    
   except KeyError:
         sAr_0, sAl_0, kl, kr, k2, k3 = params
         
   # if t >= 0 and t < 0.001:
   #      sAr = sAr_0
   #      sAl = sAl_0
   #      sB = 0
   #      sC = 0
        
   # else:
   sAr = x[0]
   sAl = x[1]
   sB =  x[2]
   sC =  x[3]
   
   dsAr_dt = -kr * sAr
   dsAl_dt = -kl * sAl
   dsB_dt  =  kl * sAl - k2 * sB
   dsC_dt  = (k2 * sB + kr * sAr) - k3 * sC
    
   return [dsAr_dt, dsAl_dt, dsB_dt, dsC_dt] 

def wrapper_model_2(t, x, params, t_compute, Rsquare, data):
   sim = scipy.integrate.solve_ivp(model_3, t, x, 'Radau', t_eval=t_compute, args=[params])
   sAr, sAl, sB, sC = sim.y
   sD = (params['sAr_0'].value + params['sAl_0'].value) - (sAr + sAl + sB + sC)
   sAt = sAr + sAl
   sol =  np.array([sAr, sAl, sB, sC, sD, sim.t, sAt])
    
   ## Calculo do R^2:
   if (Rsquare == True):
      if np.shape(data)[0] == 3:
         sAt = sAr + sAl
         R2_S = tools.r2(np.array(data[0]['concentração']), sAt)
         R2_I = tools.r2(np.array(data[1]['concentração']), sC)
         R2_P = tools.r2(np.array(data[2]['concentração']), sD)
         return [R2_S, R2_I, R2_P ]
      else:
         R2_S = tools.r2(np.array(data[0]['concentração']), sAt)
         R2_P = tools.r2(np.array(data[1]['concentração']), sD)
         return [R2_S, R2_P]
      
   else:
      return sol