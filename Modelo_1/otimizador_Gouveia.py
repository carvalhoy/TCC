import numpy as np
import models
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm

def obj_wrapper_model_2 (params, data, t, x, t_compute):
    
    sim = models.wrapper_model_2(t, x, params, t_compute, square=False, data=None)
    res = data - sim[4]
   #  res_norm = res/(max(data) - min(data))
    
    return res
 
def obj_wrapper_model_2_SP (params, data, t, x, t_compute):
    
    sim = models.wrapper_model_2(t, x, params, t_compute, square=False, data=None)
    res_SD = data[1]['concentração'] - sim[4]
    res_SA = data[0]['concentração'] - sim[6]
    
    res_total = np.sqrt(np.square(res_SA)) + np.sqrt(np.square(res_SD))
    
    return res_total