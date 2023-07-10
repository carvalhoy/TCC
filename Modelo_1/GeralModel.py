# importanto bibliotecas:
import numpy as np # 1.24.3
import scipy.integrate, scipy.optimize # 1.10.1
import matplotlib.pyplot as plt # 3.7.1
import pandas as pd # 2.0.1
from datetime import datetime as dt # 5.1
import lmfit as lm
#from lmfit import minimize, Parameters, report_fit, fit_report # 1.2.1
from simple_chalk import chalk # 0.1.0
import time  
import random


##ajustar dados utilizados na otimização:
def ajustarXlsx(caminho:str, parametrosDadosXlsx: list[int]):
    [tempoInicial, tempoFinal, totalPontos, digitosDecimais] = parametrosDadosXlsx
    dados:pd.DataFrame = pd.read_excel(caminho, header=None, names=['tempo', 'concentração'], decimal=',')
    dados['tempo'] = np.linspace(tempoInicial, tempoFinal, totalPontos)
    dados['concentração'] = round(dados['concentração']/1000, digitosDecimais)
    return dados

def model (t, x, params:lm.Parameters):
    
    S = x[0] #g_DQO_S/m^3
    X = x[1] #g_DQO_X/m^3
    P = x[2] #g_DQO_P/m^3

    try:
         ##parâmetros do modelo
        S_in = params['S_in'].value
        mu_max_X = params['mumax_X'].value #dia^-1
        K_S = params['K_S'].value #g_DQO_S/m^3
        Y_X_S = params['Y_X_S'].value #g_DQO_X/g_DQO_S
        Y_P_S = params['Y_P_S'].value #g_DQO_P/g_DQO_S
        k_dec = params['k_dec'].value #dia^-1
        D = params['D'].value #dia^-1
    
    except KeyError:
        S_in, mu_max_X, K_S, Y_X_S, Y_P_S, k_dec, D = params
    
     ##definindo a reação:
    mu = mu_max_X*S/(K_S + S) #dia^-1
    if (D == 0):
        # print('Operação em batelada')
        ##definindo balanço de componentes:
        dS_dt = - (1/Y_X_S)*mu*X
        dX_dt = (mu-k_dec)*X
        dP_dt = Y_P_S*((1/Y_X_S)*mu*X)
    
    else:
        # print('Operação contínua')
        ##definindo balanço de componentes:
        dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
        dX_dt = (D)*(-X) + (mu-k_dec)*X
        dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X)
        
        

    return [dS_dt, dX_dt, dP_dt]

def residual(paras:lm.Parameter, t_step:list[int], data:np.ndarray, t_solve_ivp:pd.Series, x:list[float], rtol:float, atol:float, indice:int, metodoIntegracao:str):
     ## resolução do modelo: 
    model = integracao(metodoIntegracao, t_step, paras, t_solve_ivp, x, rtol, atol)
     ## indexação direta com numpy ndarray
    model:np.ndarray = model.y
    modelVariaveldoModelo:np.ndarray = model[indice]
    #  ## conversão de matriz numpy em dataframe pd:
    # model = pd.DataFrame(model.y).transpose()
    #  ## definição do variável otimizada com indexação .iloc:
    # modelVariaveldoModelo = model.iloc[:, indice]
     ## cálculo do erro entre dados experimentais e o ajuste do modelo:
    eps = max(data)
    error = ((modelVariaveldoModelo - data)/eps)
    # print(f'o erro é {error}')
    return error

def residual2(paras:lm.Parameter, t_step:list[int], data:np.ndarray, t_solve_ivp:pd.Series, x:list[float], rtol:float, atol:float, metodoIntegracao:str):
     ## resolução do modelo
    model = integracao(metodoIntegracao, t_step, paras, t_solve_ivp, x, rtol, atol)
     ## indexação com numpy ndarray:
    model:np.ndarray = model.y
     ## conversão de matriz numpy em dataframe pd:
    # model = pd.DataFrame(model.y).transpose()
     ## definição da lista de resutados do modelo com indexação .iloc:
    model_S = model[0] #substrato
    model_P = model[2] #produto
     ## conversão da lista de listas de dados experimentais para dataframe pd:
    # data = pd.DataFrame(data).transpose()
    epsS = max(data[0])
    epsP = max(data[1])
     ## cálculo do erro entre dados experimentais e ajuste do modelo:
    erro_P = np.square((model_P - data[1])/epsP)
    erro_S = np.square((model_S - data[0])/epsS)
    error = (erro_P + erro_S)
    return error

def integracao(metodoIntegracao:str, t_step:list[int], params:lm.Parameters, t_solve_ivp:pd.Series, x:list[float], rtol:float, atol:float):
     ## integração do modelo com função .solve_ivp:
    solve_ivp:object = scipy.integrate.solve_ivp(model, t_step, x, metodoIntegracao, t_eval=t_solve_ivp, args=[params], rtol=rtol, atol=atol)
    return solve_ivp

    ## P/ indice = 10: 2 parametros.
def  r2(dado, indice:int, metodoIntegracao:str, t_step:list[int], params:lm.Parameters, t_solve_ivp:pd.Series, x:list[float], rtol:float, atol:float):
    dado = np.array(dado)
    otim_model = integracao(metodoIntegracao, t_step, params, t_solve_ivp, x, rtol, atol).y
    r_square: list[float] | float
    if indice == 10:
        res_S = dado[0] - otim_model[0]
        S_SSR = sum(np.square(res_S))
        S_SST = sum(np.square(dado[0] - np.mean(dado[0])))
        res_P = dado[1] - otim_model[2]
        P_SSR = sum(np.square(res_P))
        P_SST = sum(np.square(dado[1] - np.mean(dado[1])))
        r_square = [(1 - (S_SSR/S_SST)), (1 - (P_SSR/P_SST))]
        # r_square = 1 - (S_SSR + P_SSR)/(S_SST + P_SST)  
    else:
        r_square = 1-sum(np.square(dado-otim_model[indice]))/sum(np.square(dado-np.mean(dado)))  
    return r_square

def plotagem(solve_ivp:object, dados_P:pd.DataFrame, dados_S:pd.DataFrame, metodo_integracao:str, metodo_otimizacao:str, parametroOtimizado:lm.Parameters):
     ## abrindo a tupla retornada em três variáveis:
    S_solve_ivp, X_solve_ivp, P_solve_ivp = solve_ivp.y    
     ## definindo título da figura:
    plt.figure().suptitle(f'${metodo_integracao} + ${metodo_otimizacao} - {parametroOtimizado}')    
     ## plotando dados experimentais de concentração de produto:
    plt.plot(dados_P['tempo'], dados_P['concentração'], 'o', color='b', label="P exp")    
     ## plotando dados experimentais de concentração de substrato:
    plt.plot(dados_S['tempo'], dados_S['concentração'], 'o', color='r', label='S exp')     
     ## plotando resultado de S da integração do modelo:
    plt.plot(solve_ivp.t, S_solve_ivp, color='red', label='S ajuste')    
     ## plotando resultado de X da integração do modelo:
    plt.plot(solve_ivp.t, X_solve_ivp, color='green', label='X ajuste')    
     ## plotando resultado de P da integração do modelo:
    plt.plot(solve_ivp.t, P_solve_ivp, color='blue', label='P ajuste')    
     ## título eixo x:
    plt.xlabel(r"t - dias")    
     ## título eixo y:
    plt.ylabel(r"Y - $kgDQO{m^3}$")    
     ## chamada de legenda, localização da legenda fora do gráfico e tamanho da fonte:
    plt.legend(bbox_to_anchor=(1.05, 1.0) ,fontsize='8')   
     ## ajuste de espaçamento do gráfico:
    plt.tight_layout()    
     ## segurar a exibição das figuras até o final de todos os ajustes:
    plt.draw()
    
def subplts(data, parametros_S, parametros_P, parametros_S_P, metodoIntegracao , t_step, t_solve_ivp, x, rtol, atol, metodoOtimizacao):
    
    # fit_S = scipy.integrate.solve_ivp(model, [0, 240], x, metodoIntegracao, args=[tuple(parametros[0])], t_eval=t_solve_ivp)
    # fit_P = scipy.integrate.solve_ivp(model, [0, 240], x, args=[tuple(parametros[1])], t_eval=t_solve_ivp)
    # fit_S_P = scipy.integrate.solve_ivp(model, [0, 240], x, args=[tuple(parametros[2])], t_eval=t_solve_ivp)
    fit_S = integracao(metodoIntegracao, t_step, parametros_S, t_solve_ivp, x, rtol, atol)
    fit_P = integracao(metodoIntegracao, t_step, parametros_P, t_solve_ivp, x, rtol, atol)
    fit_S_P = integracao(metodoIntegracao, t_step, parametros_S_P, t_solve_ivp, x, rtol, atol)
    
    fig, axs = plt.subplots(3, 1)
    axs[0].set_prop_cycle(color=['red', 'blue', 'red', 'green', 'blue'])
    axs[0].plot(t_solve_ivp, data[0], 'x', t_solve_ivp, data[1], 'x', t_solve_ivp, np.transpose(fit_S.y))
    axs[0].legend(['S exp', 'P exp', 'S fit', 'X fit', 'P fit'], bbox_to_anchor=(1.05, 1.0), fontsize='7')
    axs[0].set_title(f'${metodoIntegracao} + ${metodoOtimizacao} - Ajuste de Substrato', fontsize=8)
    axs[0].set_ylabel(r"C - $kgDQO{m^3}$")  

    
    axs[1].set_prop_cycle(color=['red', 'blue', 'red', 'green', 'blue'])
    axs[1].plot(t_solve_ivp, data[0], 'x', t_solve_ivp, data[1], 'x', t_solve_ivp, np.transpose(fit_P.y))
    axs[1].legend(['S exp', 'P exp', 'S fit', 'X fit', 'P fit'], bbox_to_anchor=(1.05, 1.0), fontsize='7')
    axs[1].set_title(f'${metodoIntegracao} + ${metodoOtimizacao} - Ajuste de Produto', fontsize=8)
    axs[1].set_ylabel(r"C - $kgDQO{m^3}$")  
    
    axs[2].set_prop_cycle(color=['red', 'blue', 'red', 'green', 'blue'])
    axs[2].plot(t_solve_ivp, data[0], 'x', t_solve_ivp, data[1], 'x', t_solve_ivp, np.transpose(fit_S_P.y))
    axs[2].legend(['S exp', 'P exp', 'S fit', 'X fit', 'P fit'], bbox_to_anchor=(1.05, 1.0), fontsize='7')
    axs[2].set_title(f'${metodoIntegracao} + ${metodoOtimizacao} - Ajuste de S e P', fontsize=8)
    axs[2].set_ylabel(r"C - $kgDQO{m^3}$")  
    
    
def writeReport(id:str, ranges: str, resultProduto:object, resultSubstrato:object, resultGeral:object, tempo_produto:float, tempo_substrato:float, tempo_duasvar:float, coefcorr_produto:float, coefcorr_substrato:float, coefcorr_geral:float, caminho:str):
    report = open(caminho, 'a')
    report.write(f'\n\n************************************** REPORT ID {id} - {dt.now()}: **************************************\n\n************* ESPACO PARAMETRICO *************\n\n{ranges}\n\n************* PRODUTO *************\n\n{lm.fit_report(resultProduto)}\n\n************* SUBSTRATO ************* \n\n{lm.fit_report(resultSubstrato)}\n\n************* GERAL ************* \n\n{lm.fit_report(resultGeral)}')
    report.write(f'\n\n************* Tempos de execucao ************* \nProduto: {tempo_produto:.2f} s\nSubstrato: {tempo_substrato:.2f} s\nDuas variaveis: {tempo_duasvar:.2f} s')
    report.write(f'\n\n************* R-square ************* \nProduto: {coefcorr_produto:.4f}\nSubstrato: {coefcorr_substrato:.4f}\nDuas variaveis: S ({coefcorr_geral[0]:.4f}) ; P({coefcorr_geral[1]:.4f})')
    report.close()

def main():
    parametrosDadosXlsx:list[int] = [0, 240, 25, 3] #tempo inicial, tempo final, número de pontos, algarismos significativos
    ## importando dados experimentais do excel:
     ### checar caminho do arquivo###
    dados_P = ajustarXlsx("./xlsx1/produto.xlsx", parametrosDadosXlsx)
    dados_S = ajustarXlsx("./xlsx1/substrato.xlsx", parametrosDadosXlsx)
    ## condições iniciais das variáveis dos balanços: substrato, biomassa, produto;
    x:list[float] = [42.5, 25.2, 0.0]
    ## intervalo de integração:
    t_step:list[int] = [0, 240]
    ## dados no tempo a serem retornados pela função integrate.solve_ivp:
    t_solve_ivp:pd.Series = dados_P['tempo']
    ## lista de métodos de integração da função .solve_ivp:
    metodosIntegracao:list[str] = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
    ## definindo método de integração da função .solve_ivp:
    metodoIntegracao:str = metodosIntegracao[2]
    ## definindo as tolerâncias relativa e absoluta usada na função .solve_ivp:
    rtol:float = 3e-14
    atol:float = 1e-18    
    
    ## definindo parâmetros para Curve Fitting:
    paras = lm.Parameters()
    paras.add('S_in', value=0., vary=False)
    paras.add('mumax_X', value=0.05, min=0.01, max=1)
    paras.add('K_S', value=0.8, min=0.1, max=9)
    paras.add('Y_X_S', value=0.2, min=0.001, max=0.7)
    paras.add('Y_P_S', value=0.8, min=0.01, max=0.9)
    paras.add('k_dec', value=0.015, min=0.01, max=0.5)
    paras.add('D', value=0., vary=False)   
    
    ranges: str = paras.pretty_repr(oneline=False)
    ## definindo método de minimização usado na função .minimize:
    metodoMinimizacao:str = 'Nelder-Mead'    
    ## otimização para Substrato (0), Biomassa (1), Produto (2):
    indice:list[int] = [0, 1, 2]
    ## lista de listas com concentração de Substrato (0) e Produto (1)
    dadoOtimizacao:list[pd.Series] = [dados_S['concentração'], dados_P['concentração']]
    
    # ########### PRODUTO ###########
    print(chalk.green("\nminimize chamada para produto"))
    start_time_produto: float = time.time()
    resultProduto = lm.minimize(residual, paras, args=(t_step, np.array(dadoOtimizacao[1]), t_solve_ivp, x, rtol, atol, indice[2], metodoIntegracao), method=metodoMinimizacao)  
    # report_fit(resultProduto)
    resultProduto.params.pretty_print(colwidth='6', columns=['name', 'value', 'vary', 'min', 'max', 'stderr'])
    print(chalk.yellow(resultProduto.message))

    coefcorr_produto:float = r2(dadoOtimizacao[1], indice[2], metodoIntegracao, t_step, resultProduto.params, t_solve_ivp, x, rtol, atol)
    print(chalk.yellow(f'R2 = {coefcorr_produto:.4f}'))
    
    tempo_produto:float = (time.time()-start_time_produto) 
    print(chalk.yellow(f'Tempo exec: = {tempo_produto:.2f} s'))
        
    
    ########### SUBSTRATO ########### 
    print(chalk.green("\nminimize chamada para substrato"))
    start_time_substrato = time.time()
    resultSubstrato = lm.minimize(residual, paras, args=(t_step, np.array(dadoOtimizacao[0]), t_solve_ivp, x, rtol, atol, indice[0], metodoIntegracao), method=metodoMinimizacao)  
    resultSubstrato.params.pretty_print(colwidth='6', columns=['name', 'value', 'vary', 'min', 'max', 'stderr'])
    print(chalk.yellow(resultSubstrato.message))
    
    coefcorr_substrato = r2(dadoOtimizacao[0], indice[0], metodoIntegracao, t_step, resultSubstrato.params, t_solve_ivp, x, rtol, atol)
    print(chalk.yellow(f'R2 = {coefcorr_substrato:.4f}'))
    
    tempo_substrato = (time.time()-start_time_substrato)
    print(chalk.yellow(f'Tempo exec: = {tempo_substrato:.2f} s'))
    
    ###########  PRODUTO E SUBSTRATO ########### 
    print(chalk.green("\nminimize chamada para duas variáveis"))
    start_time_duasvar = time.time()
    resultGeral = lm.minimize(residual2, paras, args=(t_step, np.array(dadoOtimizacao), t_solve_ivp, x, rtol, atol, metodoIntegracao), method=metodoMinimizacao)  # leastsq
    resultGeral.params.pretty_print(colwidth='6', columns=['name', 'value', 'vary', 'min', 'max', 'stderr'])
    print(chalk.yellow(resultGeral.message))
    coefcorr_geral = r2(dadoOtimizacao, 10, metodoIntegracao, t_step, resultGeral.params, t_solve_ivp, x, rtol, atol)
    print(chalk.yellow(f'R2 - S = {coefcorr_geral[0]:.4f} ; P = {coefcorr_geral[1]:.4f}'))
    tempo_duasvar = (time.time()-start_time_duasvar)
    print(chalk.yellow(f'Tempo exec: = {tempo_duasvar:.2f} s'))
    
    # REPORT DATA:
    id = str(random.randint(1,201))
    print(chalk.red('\nEscrevendo relatorio:'))
    caminho = './Modelo_1/Report.txt'
    writeReport(id, ranges, resultProduto, resultSubstrato, resultGeral, tempo_produto, tempo_substrato, tempo_duasvar, coefcorr_produto, coefcorr_substrato, coefcorr_geral, caminho)
    print(chalk.blue('OK'))

    ## Plotagem em subplots
    print(chalk.red("\nPlotagem chamada"))
    subplts(dadoOtimizacao, resultSubstrato.params, resultProduto.params, resultGeral.params, metodoIntegracao, t_step, t_solve_ivp, x, rtol, atol, metodoMinimizacao)
    plt.xlabel(r"t - dias")    
    plt.tight_layout()
    plt.savefig(f'./Modelo_1/graphs/{id}.png')
    print(chalk.blue("OK\n"))
    plt.show()
    
    
main()