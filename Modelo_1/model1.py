import lmfit as lm

def model (t, x, params:lm.Parameters):
    
   S = x[0] #kg_DQO_S/m^3
   X = x[1] #kg_DQO_X/m^3
   P = x[2] #kg_DQO_P/m^3
   
        
   try:
      ##parâmetros do modelo
      S_in = params['S_in'].value #kg_DQO_S/m3
      mu_max_X = params['mumax_X'].value #dia^-1
      K_S = params['K_S'].value #kg_DQO_S/m^3
      Y_X_S = params['Y_X_S'].value #kg_DQO_X/kg_DQO_S
      Y_P_S = params['Y_P_S'].value #kg_DQO_P/kg_DQO_S
      k_dec = params['k_dec'].value #dia^-1
      D = params['D'].value #dia^-1
        
    
   except KeyError:
       S_in, mu_max_X, K_S, Y_X_S, Y_P_S, k_dec, D = params
    
   ##definindo a reação:
   mu = mu_max_X*S/(K_S + S) #dia^-1
   if (D == 0):
        # print('Operação em batelada')
        ##definindo balanço de componentes:
        dS_dt = - (mu/Y_X_S)*X
        dX_dt = (mu-k_dec)*X
        dP_dt = ((Y_P_S/Y_X_S)*mu*X)
    
   else:
        # print('Operação contínua')
        ##definindo balanço de componentes:
        dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
        dX_dt = (D)*(-X) + (mu-k_dec)*X
        dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X)
        
        

   return [dS_dt, dX_dt, dP_dt]