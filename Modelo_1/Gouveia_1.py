import lmfit as lm
import matplotlib.pyplot as plt
import tools
import models

# Dados experimentais:
parametrosDadosXlsx:list[int] = [0, 240, 25, 3]
P_exp = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_produto.csv", parametrosDadosXlsx)
S_exp = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_substrato.csv", parametrosDadosXlsx)
data = [S_exp, P_exp]

## Parâmetros do modelo otimizado de Gouveia et al. (2022):
paras = lm.Parameters()
paras.add('S_max', value=38.1, vary=False) # gCOD/L
paras.add('Kl', value=0.0133, vary=False) # dia^-1
paras.add('Kr', value=0.1532, vary=False) # dia^-1
paras.add('K2', value=0.1274, vary=False) # dia^-1
paras.add('K3', value=0.1181, vary=False) # dia^-1
paras.add('alpha', value=0.3532, vary=False) # [-]

# Resolução:
parametros_Iniciais = [38.1, 0.,0.]
sim = models.wrapper_model([0, 240], parametros_Iniciais, paras, None, False, None)
sA, sB, sC, sD, t_sim = sim
R2_S_P = models.wrapper_model([0, 240], parametros_Iniciais, paras, P_exp['tempo'], True, data)

# Plotagem:
plt.plot(t_sim, sA, 'b', label=f'$S_A (R^2 = {R2_S_P[0]:.3f})$')
plt.plot(t_sim, sB, 'y', label='$S_B$')
plt.plot(t_sim, sC, 'g', label='$S_C$')
plt.plot(t_sim, sD, 'r', label=f'$S_D (R^2 = {R2_S_P[1]:.3f})$')
plt.plot(P_exp['tempo'], P_exp['concentração'], 'rx', label='$S_D exp.$' )
plt.plot(S_exp['tempo'], S_exp['concentração'], 'bx', label='$S_A exp.$' )
plt.xlabel('t - dias', fontsize=15)    
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.legend(fontsize=15)
plt.show()
