#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
from scipy.stats import norm
import numpy as np
import pandas as pd

def events_rate(x, y):
    return (y + 0.0) / x

def calc_nonevent(x):
    return x[0] - x[1]

def woe_calc(x, target_name, var_name, ascending_f, n_threshold, p_threshold):
    
    #Разделили датасет на две части: NULL и NOTNULL
    woe_null = x[x[var_name].isnull()]
    woe_notnull = x[x[var_name].notnull()]
    
    #Для каждого значения независимой переменной набора NOTNULL считаем кол-во записей и Event'ов
    woe = woe_notnull.groupby(var_name).agg({target_name : ['count', 'sum', 'std']})
    woe.columns = woe.columns.droplevel(level = 0)
    woe['events_rate'] = woe.apply(lambda x: events_rate(x['count'], x['sum']), axis =  1)
    woe = woe.reset_index()
    woe[var_name + '_min'] = woe[var_name]
    woe[var_name + '_max'] = woe[var_name]
    
    # После сортировки, сначала должны идти значения с наибольшей долей Events (!!!по логике!!!)
    woe.sort_values(by = [var_name], ascending = ascending_f, inplace = True)
    woe = woe.reset_index(drop = True)
   
    
    #Merge бакетов по Events Rate
    while True:
        break_f = True
        cnt = woe.shape[0]
        
        if cnt <= 1:
            break
        
        cur_i = 0
        next_i = 1
        
        while True:
            if (cnt <= 1) | (cur_i == (cnt - 1)):
                break
            
            cur_rate = woe['events_rate'].loc[cur_i]
            next_rate = woe['events_rate'].loc[next_i]
            
            if (cur_rate > next_rate):
                cur_i = next_i
                next_i = cur_i + 1
                continue
            else:
                break_f = False
                
                if (cur_i == 0):
                    j = next_i
                else:
                    prev_rate = woe['events_rate'].loc[cur_i - 1]
                    j = next_i
                
                count2 = woe['count'].loc[cur_i] + woe['count'].loc[j]
                sum2 = woe['sum'].loc[cur_i] + woe['sum'].loc[j]
                events_rate2 = (sum2 + 1.0) / count2
                std2 = math.sqrt((sum2 * (1 - 2 * events_rate2) + count2 * events_rate2**2) / (count2 - 1))
                
                woe['count'].loc[j] = count2
                woe['sum'].loc[j] = sum2
                woe['std'].loc[j] = std2
                woe['events_rate'].loc[j] = events_rate2
                
                woe[var_name + '_min'].loc[j] = min(woe[var_name + '_min'].loc[cur_i],                                                          woe[var_name + '_min'].loc[j])
                woe[var_name + '_max'].loc[j] = max(woe[var_name + '_max'].loc[cur_i],                                                          woe[var_name + '_max'].loc[j])
                woe.drop([cur_i], inplace = True)
                woe = woe.reset_index(drop = True)
                cnt = woe.shape[0]
                
                if (cur_i > 0):
                    cur_i = cur_i - 1
                    next_i = next_i - 1
                continue
                
        if (break_f == True):
            break
            
    #Merge бакетов по P-Value
    while True:
        if cnt <= 1:
            break
            
        pval_list = []
        count_list = []
        event_list = []
        std_list = []
        
        for i in woe.index:
            if i == 0:
                continue
            
            count_list.append(woe['count'].loc[i] + woe['count'].loc[i - 1])
            event_list.append(woe['sum'].loc[i] + woe['sum'].loc[i - 1])
            events_rate2 = (event_list[i - 1] + 1.0) / count_list[i - 1]
            
            std_list.append((event_list[i - 1] * (1 - 2 * events_rate2) +
                             count_list[i - 1] * events_rate2**2) / (count_list[i - 1] - 2))
            
            if (std_list[i - 1] > 0):
                zval = (woe['events_rate'].loc[i - 1] - woe['events_rate'].loc[i] + 0.0) /                        math.sqrt(std_list[i - 1] * (1.0 / woe['count'].loc[i - 1] + 1.0 / woe['count'].loc[i]))
                pval = 1 - norm.cdf(zval)
            else:
                pval = 2
            
            if (woe['count'].loc[i - 1] < n_threshold) | (woe['count'].loc[i] < n_threshold):
                pval = pval + 1
            pval_list.append(pval)
        
        if(max(pval_list) < p_threshold):
            break
        else:
            ind_max = pval_list.index(max(pval_list))
            woe['count'].loc[ind_max] = count_list[ind_max]
            woe['sum'].loc[ind_max] = event_list[ind_max]
            woe['std'].loc[ind_max] = std_list[ind_max] * (count_list[ind_max] - 2) / (count_list[ind_max] - 1)
            woe['events_rate'].loc[ind_max] = (event_list[ind_max] + 1.0) / count_list[ind_max]
            
            woe[var_name + '_min'].loc[ind_max] = min(woe[var_name + '_min'].loc[ind_max],                                                          woe[var_name + '_min'].loc[ind_max + 1])
            woe[var_name + '_max'].loc[ind_max] = max(woe[var_name + '_max'].loc[ind_max],                                                      woe[var_name + '_max'].loc[ind_max + 1])
            
            woe.drop([ind_max + 1], inplace = True)
            woe = woe.reset_index(drop = True)
            cnt = woe.shape[0]
    
    woe['NONEVENT'] = woe[['count', 'sum']].apply(lambda x: calc_nonevent(x), axis = 1)
    woe.rename(columns = {var_name + '_min': 'MIN_VALUE', var_name + '_max': 'MAX_VALUE', 'sum': 'EVENT'}, inplace = True)
    woe1 = woe[['MIN_VALUE', 'MAX_VALUE', 'EVENT', 'NONEVENT']]
    
    ########################   РАБОТАЕМ С NULL   ########################
    if woe_null.shape[0] > 0:
        cnt_event = 0
        cnt_nonevent = 0
        
        for j in woe_null.index:
            cnt_event = cnt_event + woe_null[target_name].loc[j]
            cnt_nonevent = cnt_nonevent + (1 if woe_null[target_name].loc[j] == 0 else 0)
        
        woe2 = pd.DataFrame({'MIN_VALUE':[np.NaN], 'MAX_VALUE':[np.NaN], 'EVENT':cnt_event, 'NONEVENT':cnt_nonevent})
        woe_final = pd.concat([woe1, woe2])
    else:
        woe_final = woe1
    
    woe_final = woe_final.reset_index(drop = True)
    woe_final['WOE'] = 0
    iv = 0
    
    for i in woe_final.index:
        woe_final['WOE'].loc[i] = np.log((woe_final['EVENT'].loc[i] + 0.0) / woe_final['NONEVENT'].loc[i]) -                             np.log((woe_final['EVENT'].sum() + 0.0) / woe_final['NONEVENT'].sum())
        iv = iv + ((woe_final['EVENT'].loc[i] + 0.0) / woe_final['EVENT'].sum() -                    (woe_final['NONEVENT'].loc[i] + 0.0) / woe_final['NONEVENT'].sum()) *                     woe_final['WOE'].loc[i]
    
    woe_final['IV'] = iv
    woe_final['VAR_NAME'] = var_name
    
    return woe_final

