import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import tools
import models

# Dados experimentais:
parametrosDadosXlsx:list[int] = [0, 240, 25, 3]
data_fit_P = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_produto.csv", parametrosDadosXlsx)
data_fit_S = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_substrato.csv", parametrosDadosXlsx)
data = [data_fit_S, data_fit_P]
t = data_fit_P['tempo']
t_plot = np.linspace(min(t), max(t), 1000)

## Parâmetros do modelo otimizado de Gouvea et al. (2022):
paras = lm.Parameters()
paras.add('S_max', value=38.1, vary=False) # gCOD/L
paras.add('Kl', value=0.0133, vary=False) # dia^-1
paras.add('Kr', value=0.1532, vary=False) # dia^-1
paras.add('K2', value=0.1274, vary=False) # dia^-1
paras.add('K3', value=0.1181, vary=False) # dia^-1
paras.add('alfa', value=0.3532, vary=False) # [-]

resolucao = models.modelo_analitico(paras, t_plot, False, None) #[sA, Sb, Sc, Sd]
    
R2_S_P = models.modelo_analitico(paras, t, True, data)

plt.plot(t, data_fit_P['concentração'], 'rx', label=f'$S_D exp.$')
plt.plot(t, data_fit_S['concentração'], 'bx', label=f'$S_A exp.$')
plt.plot(t_plot, resolucao[0], 'b', label=f'$S_A (R^2={R2_S_P[0]:.3f})$')
plt.plot(t_plot, resolucao[1], 'y', label='$S_B$')
plt.plot(t_plot, resolucao[2], 'g', label='$S_C$')
plt.plot(t_plot, resolucao[3], 'r', label=f'$S_D (R^2={R2_S_P[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()




    
    