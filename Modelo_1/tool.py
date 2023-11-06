import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import lmfit as lm
import simulador

##ajustar dados utilizados na otimização:
def ajustarXlsx(caminho:str, parametrosDadosXlsx: list[int]):
    [tempoInicial, tempoFinal, totalPontos, digitosDecimais] = parametrosDadosXlsx
    dados:pd.DataFrame = pd.read_csv(caminho, header=None, names=['tempo', 'concentração'])
    dados['tempo'] = np.linspace(tempoInicial, tempoFinal, totalPontos)
    dados['concentração'] = round(dados['concentração'], digitosDecimais)
    # print(dados)
    return dados
 
def subplts(data, parametros_S, parametros_P, parametros_S_P, metodoIntegracao , t_step, t_solve_ivp, x, rtol, atol, metodoOtimizacao):
    
    # fit_S = scipy.integrate.solve_ivp(model, [0, 240], x, metodoIntegracao, args=[tuple(parametros[0])], t_eval=t_solve_ivp)
    # fit_P = scipy.integrate.solve_ivp(model, [0, 240], x, args=[tuple(parametros[1])], t_eval=t_solve_ivp)
    # fit_S_P = scipy.integrate.solve_ivp(model, [0, 240], x, args=[tuple(parametros[2])], t_eval=t_solve_ivp)
    fit_S = simulador.integracao(metodoIntegracao, t_step, parametros_S, None, x, rtol, atol)
    # fit_S = scipy.integrate.solve_ivp(modelP, t_step, x, metodoIntegracao, args=[parametros_S], max_step=0.01)

    fit_P = simulador.integracao(metodoIntegracao, t_step, parametros_P, None, x, rtol, atol)
    # fit_P = scipy.integrate.solve_ivp(modelP, t_step, x, metodoIntegracao, args=[parametros_P], max_step=0.01)
    fit_S_P = simulador.integracao(metodoIntegracao, t_step, parametros_S_P, None, x, rtol, atol)
    # fit_S_P = scipy.integrate.solve_ivp(modelP, t_step, x, metodoIntegracao, args=[parametros_S_P], max_step=0.01)
    
    fig, axs = plt.subplots(3, 1)
    axs[0].set_prop_cycle(color=['red', 'blue', 'red', 'green', 'blue'])
    axs[0].plot(t_solve_ivp, data[0], 'x', t_solve_ivp, data[1], 'x', fit_S.t, np.transpose(fit_S.y))
    axs[0].legend(['S exp', 'P exp', 'S fit', 'X fit', 'P fit'], bbox_to_anchor=(1.05, 1.0), fontsize='7')
    axs[0].set_title(f'${metodoIntegracao} + ${metodoOtimizacao} - Ajuste de Substrato', fontsize=8)
    axs[0].set_ylabel(r"C - $kgDQO{m^3}$")  

    
    axs[1].set_prop_cycle(color=['red', 'blue', 'red', 'green', 'blue'])
    axs[1].plot(t_solve_ivp, data[0], 'x', t_solve_ivp, data[1], 'x', fit_P.t, np.transpose(fit_P.y))
    axs[1].legend(['S exp', 'P exp', 'S fit', 'X fit', 'P fit'], bbox_to_anchor=(1.05, 1.0), fontsize='7')
    axs[1].set_title(f'${metodoIntegracao} + ${metodoOtimizacao} - Ajuste de Produto', fontsize=8)
    axs[1].set_ylabel(r"C - $kgDQO{m^3}$")  
    
    axs[2].set_prop_cycle(color=['red', 'blue', 'red', 'green', 'blue'])
    axs[2].plot(t_solve_ivp, data[0], 'x', t_solve_ivp, data[1], 'x', fit_S_P.t, np.transpose(fit_S_P.y))
    axs[2].legend(['S exp', 'P exp', 'S fit', 'X fit', 'P fit'], bbox_to_anchor=(1.05, 1.0), fontsize='7')
    axs[2].set_title(f'${metodoIntegracao} + ${metodoOtimizacao} - Ajuste de S e P', fontsize=8)
    axs[2].set_ylabel(r"C - $kgDQO{m^3}$")  
    
    
def writeReport(id:str, ranges: str, resultProduto:object, resultSubstrato:object, resultGeral:object, tempo_produto:float, tempo_substrato:float, tempo_duasvar:float, coefcorr_produto:float, coefcorr_substrato:float, coefcorr_geral:float, caminho:str):
    report = open(caminho, 'a')
    report.write(f'\n\n************************************** REPORT ID {id} - {dt.now()}: **************************************\n\n************* ESPACO PARAMETRICO *************\n\n{ranges}\n\n************* PRODUTO *************\n\n{lm.fit_report(resultProduto)}\n\n************* SUBSTRATO ************* \n\n{lm.fit_report(resultSubstrato)}\n\n************* GERAL ************* \n\n{lm.fit_report(resultGeral)}')
    report.write(f'\n\n************* Tempos de execucao ************* \nProduto: {tempo_produto:.2f} s\nSubstrato: {tempo_substrato:.2f} s\nDuas variaveis: {tempo_duasvar:.2f} s')
    report.write(f'\n\n************* R-square ************* \nProduto: {coefcorr_produto:.4f}\nSubstrato: {coefcorr_substrato:.4f}\nDuas variaveis: S ({coefcorr_geral[0]:.4f}) ; P({coefcorr_geral[1]:.4f})')
    report.close()