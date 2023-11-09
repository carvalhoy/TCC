import pandas as pd
import numpy as np

def ajustarXlsx(caminho:str, parametrosDadosXlsx: list[int]):
    [tempoInicial, tempoFinal, totalPontos, digitosDecimais] = parametrosDadosXlsx
    dados:pd.DataFrame = pd.read_csv(caminho, header=None, names=['tempo', 'concentração'])
    dados['tempo'] = np.linspace(tempoInicial, tempoFinal, totalPontos)
    dados['concentração'] = round(dados['concentração'], digitosDecimais)
    return dados
 
def r2(data, model):
    res = data - model
    coef = 1 - np.sum(np.square(res))/np.sum(np.square(np.mean(data) - model))
    return coef
