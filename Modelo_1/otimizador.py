import lmfit as lm
import numpy as np
import pandas as pd
import simulador

def residual(paras:lm.Parameter, t_step:list[int], data:np.ndarray, t_solve_ivp:pd.Series, x:list[float], rtol:float, atol:float, indice:int, metodoIntegracao:str):
     ## resolução do modelo: 
    model = simulador.integracao(metodoIntegracao, t_step, paras, t_solve_ivp, x, rtol, atol)
     ## indexação direta com numpy ndarray
    model:np.ndarray = model.y
    modelVariaveldoModelo:np.ndarray = model[indice]
    #  ## conversão de matriz numpy em dataframe pd:
    # model = pd.DataFrame(model.y).transpose()
    #  ## definição do variável otimizada com indexação .iloc:
    # modelVariaveldoModelo = model.iloc[:, indice]
     ## cálculo do erro entre dados experimentais e o ajuste do modelo:
    eps = max(data) - min(data)
    error = ((modelVariaveldoModelo - data)/eps)
    # print(f'o erro é {error}')
    return error

def residual2(paras:lm.Parameter, t_step:list[int], data:np.ndarray, t_solve_ivp:pd.Series, x:list[float], rtol:float, atol:float, metodoIntegracao:str):
     ## resolução do modelo
    model = simulador.integracao(metodoIntegracao, t_step, paras, t_solve_ivp, x, rtol, atol)
     ## indexação com numpy ndarray:
    model:np.ndarray = model.y
     ## conversão de matriz numpy em dataframe pd:
    # model = pd.DataFrame(model.y).transpose()
     ## definição da lista de resutados do modelo com indexação .iloc:
    model_S = model[0] #substrato
    model_P = model[2] #produto
     ## conversão da lista de listas de dados experimentais para dataframe pd:
    # data = pd.DataFrame(data).transpose()
    epsS = max(data[0]) - min(data[0])
    epsP = max(data[1]) - min(data[1])
     ## cálculo do erro entre dados experimentais e ajuste do modelo:
    erro_P = np.square((model_P - data[1])/epsP)
    erro_S = np.square((model_S - data[0])/epsS)
    error = (erro_P + erro_S)
    return error
 
def r2(dado, indice:int, metodoIntegracao:str, t_step:list[int], params:lm.Parameters, t_solve_ivp:pd.Series, x:list[float], rtol:float, atol:float):
    dado = np.array(dado)
    otim_model = simulador.integracao(metodoIntegracao, t_step, params, t_solve_ivp, x, rtol, atol).y
    r_square: list[float] | float
    if indice == 10:
        res_S = dado[0] - otim_model[0]
        S_SSR = sum(np.square(res_S))
        S_SST = sum(np.square(otim_model[0] - np.mean(dado[0])))
        res_P = dado[1] - otim_model[2]
        P_SSR = sum(np.square(res_P))
        P_SST = sum(np.square(otim_model[2] - np.mean(dado[1])))
        r_square = [(1 - (S_SSR/S_SST)), (1 - (P_SSR/P_SST))]
        # r_square = 1 - (S_SSR + P_SSR)/(S_SST + P_SST)  
    else:
        r_square = 1-sum(np.square(dado-otim_model[indice]))/sum(np.square(otim_model[indice]-np.mean(dado)))  
    return r_square