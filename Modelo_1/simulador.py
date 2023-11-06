import scipy.integrate
import pandas as pd
import lmfit as lm
import model1

def integracao(metodoIntegracao:str, t_step:list[int], params:lm.Parameters, t_solve_ivp:pd.Series, x:list[float], rtol:float, atol:float):
     ## integração do modelo com função .solve_ivp:
    solve_ivp:object = scipy.integrate.solve_ivp(model1.model, t_step, x, metodoIntegracao, t_eval=t_solve_ivp, args=[params], rtol=rtol, atol=atol)
    return solve_ivp