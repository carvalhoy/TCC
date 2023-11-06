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
import tool 
import simulador
import otimizador

parametrosDadosXlsx:list[int] = [0, 240, 25, 3] #tempo inicial, tempo final, número de pontos, algarismos significativos

## importando dados experimentais do excel:
### checar caminho do arquivo###
dados_P = tool.ajustarXlsx("../xlsx1/dados_gouveia_rao_produto.csv", parametrosDadosXlsx)
dados_S = tool.ajustarXlsx("../xlsx1/dados_gouveia_rao_substrato.csv", parametrosDadosXlsx)

## condições iniciais das variáveis dos balanços: substrato, biomassa, produto;
x:list[float] = [38.162, 25.2, 0.0]
## intervalo de integração:
t_step:list[int] = [0, 240]
## dados no tempo a serem retornados pela função integrate.solve_ivp:
t_solve_ivp:pd.Series = dados_P['tempo']
## lista de métodos de integração da função .solve_ivp:
metodosIntegracao:list[str] = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
## definindo método de integração da função .solve_ivp:
metodoIntegracao:str = metodosIntegracao[3]
## definindo as tolerâncias relativa e absoluta usada na função .solve_ivp:
rtol:float = 1e-8
atol:float = 1e-10    

## definindo parâmetros para Curve Fitting:
paras = lm.Parameters()
paras.add('S_in', value=0., vary=False) #kgDQO_S/m3
paras.add('mumax_X', value=0.05, min=0.02, max=0.8) #dia-1
paras.add('K_S', value=125.0, min=40.3, max=403.) #kgDQO_S/m3
paras.add('Y_X_S', value=0.07, min=0.01, max=1-0.877) #kgDQO_X/kgDQO_S
paras.add('Y_P_S', value=0.877, vary=False) #kgDQO_P/kgDQO_S
paras.add('k_dec', value=0.001755, min=0.0005, max=0.035) #dia-1
paras.add('D', value=0., vary=False) #dia-1   

ranges: str = paras.pretty_repr(oneline=False)

## definindo método de minimização usado na função .minimize:
metodoMinimizacao:str = 'Nelder-Mead'    
## otimização para Substrato (0), Biomassa (1), Produto (2):
indice:list[int] = [0, 1, 2]

## lista de listas com concentração de Substrato (0) e Produto (1)
dadoOtimizacao:list[pd.Series] = [dados_S['concentração'], dados_P['concentração']]

########### PRODUTO ###########
print(chalk.green("\nminimize chamada para produto"))
start_time_produto: float = time.time()
resultProduto = lm.minimize(otimizador.residual, paras, args=(t_step, np.array(dadoOtimizacao[1]), t_solve_ivp, x, rtol, atol, indice[2], metodoIntegracao), method=metodoMinimizacao)  
# report_fit(resultProduto)
resultProduto.params.pretty_print(colwidth='6', columns=['name', 'value', 'vary', 'min', 'max', 'stderr'])
print(chalk.yellow(resultProduto.message))
print(list(resultProduto.params.valuesdict().values()))


coefcorr_produto:float = otimizador.r2(dadoOtimizacao[1], indice[2], metodoIntegracao, t_step, resultProduto.params, t_solve_ivp, x, rtol, atol)
print(chalk.yellow(f'R2 = {coefcorr_produto:.4f}'))

tempo_produto:float = (time.time()-start_time_produto) 
print(chalk.yellow(f'Tempo exec: = {tempo_produto:.2f} s'))

########### SUBSTRATO ########### 
print(chalk.green("\nminimize chamada para substrato"))
start_time_substrato = time.time()
resultSubstrato = lm.minimize(otimizador.residual, paras, args=(t_step, np.array(dadoOtimizacao[0]), t_solve_ivp, x, rtol, atol, indice[0], metodoIntegracao), method=metodoMinimizacao)  
resultSubstrato.params.pretty_print(colwidth='6', columns=['name', 'value', 'vary', 'min', 'max', 'stderr'])
print(chalk.yellow(resultSubstrato.message))

coefcorr_substrato = otimizador.r2(dadoOtimizacao[0], indice[0], metodoIntegracao, t_step, resultSubstrato.params, t_solve_ivp, x, rtol, atol)
print(chalk.yellow(f'R2 = {coefcorr_substrato:.4f}'))

tempo_substrato = (time.time()-start_time_substrato)
print(chalk.yellow(f'Tempo exec: = {tempo_substrato:.2f} s'))

###########  PRODUTO E SUBSTRATO ########### 
print(chalk.green("\nminimize chamada para duas variáveis"))
start_time_duasvar = time.time()
resultGeral = lm.minimize(otimizador.residual2, paras, args=(t_step, np.array(dadoOtimizacao), t_solve_ivp, x, rtol, atol, metodoIntegracao), method=metodoMinimizacao)  # leastsq
resultGeral.params.pretty_print(colwidth='6', columns=['name', 'value', 'vary', 'min', 'max', 'stderr'])
print(chalk.yellow(resultGeral.message))
coefcorr_geral = otimizador.r2(dadoOtimizacao, 10, metodoIntegracao, t_step, resultGeral.params, t_solve_ivp, x, rtol, atol)
print(chalk.yellow(f'R2 - S = {coefcorr_geral[0]:.4f} ; P = {coefcorr_geral[1]:.4f}'))
tempo_duasvar = (time.time()-start_time_duasvar)
print(chalk.yellow(f'Tempo exec: = {tempo_duasvar:.2f} s'))

# REPORT DATA:
id = str(random.randint(1,201))
print(chalk.red('\nEscrevendo relatorio:'))
caminho = './Report.txt'
tool.writeReport(id, ranges, resultProduto, resultSubstrato, resultGeral, tempo_produto, tempo_substrato, tempo_duasvar, coefcorr_produto, coefcorr_substrato, coefcorr_geral, caminho)
print(chalk.blue('OK'))

## Plotagem em subplots
print(chalk.red("\nPlotagem chamada"))
tool.subplts(dadoOtimizacao, resultSubstrato.params, resultProduto.params, resultGeral.params, metodoIntegracao, t_step, t_solve_ivp, x, rtol, atol, metodoMinimizacao)
plt.xlabel(r"t - dias")    
plt.tight_layout()
plt.savefig(f'./graphs/{id}.png')
print(chalk.blue("OK\n"))
plt.show()