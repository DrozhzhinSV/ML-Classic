import numpy as np
import pandas as pd
import random
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import *





# ----------   Инициализация генетического алгоритма   ----------

#***   INPUT   ***: 
# Количество родителей - INT, 
# Гиперпараметры       - DICT. 

#В словаре указываем гиперпараметры, которые будем настраивать и мин./макс. возможное значение каждого гиперпараметра, а также шаг для определения значений

#***   OUTPUT   ***: 
# Словарь родителей - DICT(ключ - имя родителя, значение - numpy массив с гиперпараметрами в лексикографическом порядке)

def initilialize_poplulation(numberOfParents, hyperparameterSettings):

    # ИНИЦИАЛИЗИРУЕМ РОДИТЕЛЕЙ
    for i in list(sorted(hyperparameterSettings.keys())):
            
        if(type(hyperparameterSettings[i][0]) == int):
            locals()[i] = np.array(random.choices(range(hyperparameterSettings[i][0], 
                                                        hyperparameterSettings[i][1], 
                                                        hyperparameterSettings[i][2]), 
                                                  k = numberOfParents)).reshape(numberOfParents, 1)
                
        else:
            locals()[i] = np.array(random.choices(np.round(np.arange(hyperparameterSettings[i][0], 
                                                                     hyperparameterSettings[i][1], 
                                                                     step = hyperparameterSettings[i][2]), 3), 
                                                           k = numberOfParents)).reshape(numberOfParents, 1)       
        
    
    population = np.empty([numberOfParents, 1])
    for i in list(sorted(hyperparameterSettings.keys())):
        population = np.concatenate((population, locals()[i]), axis = 1)
        
    
    # ПЕЧАТАЕМ ДАННЫЕ О РОДИТЕЛЯХ
    print(pd.DataFrame(population[:, 1 : ], columns = list(sorted(hyperparameterSettings.keys())), 
                       index = ['Parents ' + str(i + 1) for i in range(numberOfParents)]))
    
    populationDict = {}
    for i in range(numberOfParents):
        populationDict['Parents ' + str(i + 1)] = population[i, 1 : ]   
        
    return populationDict







# ----------   Построение модели для каждого родителя и оценка качества   ----------

#***   INPUT   ***: 
# Словарь родителей текущего поколения - DICT(ключ - имя родителя, значение - numpy массив с гиперпараметрами в лексикографическом порядке) 
# Список гиперпараметров                         - LIST,
# Тренировочная и тестовая выборка               - PANDAS DATAFRAME
# Основная метрика для сравнения гиперпараметров - STR
# Список метрик, которые хотим отслеживать       - LIST
# Коэффициент "регуляризации" для расчета кооректированного ROC AUC - FLOAT
# Значение CUTOFF для расчета метрик             - FLOAT

#***   OUTPUT   ***: 
# Словарь с метриками - DICT(ключ - имя родителя, значение - numpy массив с метриками в лексикографическом порядке)

def train_population(population, 
                     hyperparameterSettings,
                     x_train, y_train, 
                     x_test, y_test,
                     metric, metricList,
                     alg,
                     coef_adj = 10.0,
                     cutoff = 0.5,
                     cat_feat = []):
    
    hyperparameters = list(sorted(hyperparameterSettings.keys()))
    
    metricListCopy = metricList[:]
    if(metric not in metricListCopy): metricListCopy.append(metric)
    
    scoreDict = {}
    for i in list(sorted(population.keys())):
        
        if (alg == 'XGB'):
            params = {'objective': 'binary:logistic',
                      'tree_method': 'hist'}
        elif (alg == 'LightGBM'):
            params = {'objective': 'binary', 
                      'metric': 'logloss'}
        else:
            params = {'loss_function': 'Logloss',
                      'eval_metric': 'AUC'}
            
             
        num_j = 0
        for j in hyperparameters:
            if(type(hyperparameterSettings[j][0]) == int):
                params[j] = int(population[i][num_j])
            else:
                params[j] = population[i][num_j]
            num_j += 1
        
        if (alg == 'XGB'):
            model = xgb.XGBClassifier(n_jobs = -1)
            model.set_params(**params, random_state = 121)
            model.fit(x_train, y_train, verbose = False)
        elif (alg == 'LightGBM'):
            if ('num_leaves' not in hyperparameters):
                if ('max_depth' in hyperparameters):
                    params['num_leaves'] = 2**params['max_depth']
                else:
                    params['num_leaves'] = 32
            model = lgb.LGBMClassifier(n_jobs = -1)
            model.set_params(**params, random_state = 121)
            model.fit(x_train, y_train, verbose = False, categorical_feature = cat_feat) 
        else:
            train_data = cb.Pool(data = x_train,
                                 label = y_train,
                                 cat_features = cat_feat)
            
            model = cb.CatBoostClassifier()
            model.set_params(**params, random_state = 121)
            model.fit(train_data, verbose = False)
            #model.fit(x_train, y_train, verbose = False, cat_features = cat_feat) 

        
        probs_train = model.predict_proba(x_train)[:, 1]
        probs_test  = model.predict_proba(x_test)[:, 1]
        
        prediction_train = (probs_train > cutoff).astype(int)
        prediction_test = (probs_test > cutoff).astype(int)
        
        trainList = []
        testList = []
        for k in sorted(metricListCopy): 
            if(k == 'roc_auc'):
                trainList.append(globals()[k + '_score'](y_train, probs_train))
                testList.append(globals()[k + '_score'](y_test, probs_test))
            elif(k == 'roc_auc_adj'):
                trainMetric = roc_auc_score(y_train, probs_train)
                testMetric = roc_auc_score(y_test, probs_test)
                trainList.append(trainMetric - (trainMetric - testMetric) / coef_adj)
                testList.append(testMetric - (trainMetric - testMetric) / coef_adj)               
            else:
                trainList.append(globals()[k + '_score'](y_train, prediction_train))
                testList.append(globals()[k + '_score'](y_test, prediction_test))
            
        scoreDict[i] = [trainList, testList]

    # Вывод на печать результатов   
    tempDF = pd.DataFrame(index = list(sorted(scoreDict.keys())), 
                          columns = ['Train ' + i for i in sorted(metricListCopy)] +\
                                    ['Test ' + i for i in sorted(metricListCopy)])

    for i in list(sorted(scoreDict.keys())):
        k = 0
        for j in sorted(metricListCopy):
            tempDF.loc[i, 'Train ' + j] = scoreDict[i][0][k]
            tempDF.loc[i, 'Test ' + j] = scoreDict[i][1][k]
            k += 1

    print(tempDF)
    
        
    return scoreDict






# ----------   Выбор лучших родителей   ----------

#***   INPUT   ***: 
# Словарь с метриками - DICT(ключ - имя родителя, значение - numpy массив с метриками в лексикографическом порядке) 
# Количество родителей, которое нужно отобрать   - INT,
# Основная метрика для сравнения гиперпараметров - STR
# Список метрик, которые хотим отслеживать       - LIST

#***   OUTPUT   ***: 
# Список лучших родителей - LIST

def new_parents_selection(scorePopulation, numberOfParentsMating, metricList, metric):
    
    metricListCopy = metricList[:]
    if(metric not in metricListCopy): metricListCopy.append(metric)
        
    parentsDict = {}
    for i in list(sorted(scorePopulation.keys())):
        k = 0
        for j in sorted(metricListCopy):
            if(j == metric):
                parentsDict[i] = scorePopulation[i][1][k]
                break
            k += 1
    

    for j in [i[0] for i in (sorted(parentsDict.items(), key = lambda x: x[1], reverse = True))[0 : numberOfParentsMating]]:
        print(j, end = ', ')
    
    return [i[0] for i in (sorted(parentsDict.items(), key = lambda x: x[1], reverse = True))[0 : numberOfParentsMating]]







# ----------   Скрещивание родителей   ----------

#***   INPUT   ***: 
# Список лучших родителей - LIST
# Словарь родителей текущего поколения - DICT(ключ - имя родителя, значение - numpy массив с гиперпараметрами в лексикографическом порядке)
# Количество детей - INT

#***   OUTPUT   ***: 
# Словарь детей - DICT(ключ - имя ребенка, значение - numpy массив с гиперпараметрами в лексикографическом порядке)

def crossover_uniform(bestParents, population, numberOfChildren, generation):
    
    children = {}
    
    for i in range(numberOfChildren):
        parent1_index = random.choices(range(len(bestParents)))[0]
        parent2_index = random.choices(list(set(range(len(bestParents))) - {parent1_index}))[0]
        
        gypepList = []
        for gp1, gp2 in zip(population[bestParents[parent1_index]], population[bestParents[parent2_index]]):
            gypepList.append(random.choices([gp1, gp2])[0])
            
        children['Children ' + str(i + 1) + ' (gen' + str(generation) + ')'] = np.array(gypepList)
          
    return children






# ----------   Мутация детей   ----------

#***   INPUT   ***: 
# Словарь детей - DICT(ключ - имя ребенка, значение - numpy массив с гиперпараметрами в лексикографическом порядке)
# Список гиперпараметров                         - LIST,
# Кол-во гиперпараметров для мутации             - INT,

def mutation(children, hyperparameterSettings, hyperChangeCnt):
        
    hyperparameter = list(sorted(hyperparameterSettings.keys()))
    
    if(hyperChangeCnt > len(hyperparameter)):
        hyperChangeCnt = len(hyperparameter)    
    
    for i in list(children.keys()):
        
        numHyperList = list(range(len(children[i])))
        for j in range(hyperChangeCnt):
            num_hyper = random.choices(numHyperList)[0]
            numHyperList.remove(num_hyper)

            if(type(hyperparameterSettings[hyperparameter[num_hyper]][0]) == int):
                children[i][num_hyper] = children[i][num_hyper] +\
                    np.array(random.choices(range(int(hyperparameterSettings[hyperparameter[num_hyper]][0] -\
                                                            children[i][num_hyper]), 
                                                  int(hyperparameterSettings[hyperparameter[num_hyper]][1] -\
                                                            children[i][num_hyper]), 
                                                  hyperparameterSettings[hyperparameter[num_hyper]][2]), 
                                                  k = 1))

            else:
                children[i][num_hyper] = children[i][num_hyper] +\
                    np.array(random.choices(np.round(np.arange(hyperparameterSettings[hyperparameter[num_hyper]][0] -\
                                                                     children[i][num_hyper], 
                                                               hyperparameterSettings[hyperparameter[num_hyper]][1] -\
                                                                     children[i][num_hyper], 
                                                               step = hyperparameterSettings[hyperparameter[num_hyper]][2]), 3), 
                                                               k = 1))
            
            
            
            

            

# ----------   Получение нового поколения   ----------

#***   INPUT   ***: 
# Словарь родителей текущего поколения - DICT(ключ - имя родителя, значение - numpy массив с гиперпараметрами в лексикографическом порядке)
# Словарь детей - DICT(ключ - имя ребенка, значение - numpy массив с гиперпараметрами в лексикографическом порядке)
# Список лучших родителей - LIST
# Список гиперпараметров  - LIST,

#***   OUTPUT   ***: 
# Словарь родителей нового поколения - DICT(ключ - имя родителя, значение - numpy массив с гиперпараметрами в лексикографическом порядке)

def getNewPopulation(population, children, bestParents, hyperparameters):
    newPopulation = {}
    
    for i in bestParents:
        newPopulation[i] = population[i]
        
    for i in children.keys():
        newPopulation[i] = children[i]
        
    # Вывод на печать результатов   
    tempDF = pd.DataFrame(index = list(sorted(newPopulation.keys())), columns = hyperparameters)

    for i in list(sorted(newPopulation.keys())):
        k = 0
        for j in hyperparameters:
            tempDF.loc[i, j] = newPopulation[i][k]
            k += 1

    print(tempDF)
        
    return newPopulation 







# ----------   Печать на экран лучшего алгоритам   ----------

#***   INPUT   ***: 
# Название лучшего алгоритма - STR
# Массив рассчитанных метрик для лучшего алгоритма - NUMPY ARRAY
# Основная метрика для сравнения гиперпараметров - STR
# Список метрик, которые хотим отслеживать       - LIST
# Список гиперпараметров                         - LIST,
# Массив значений гиперпараметров - NUMPY ARRAY

def printBestAlgorithm(bestParents, scorePopulation, metricList, metric, hyperparameter, population):
    print('\n\n\n\n')
    print(f"Best Algorithm: {bestParents}")
    
    metric_train = scorePopulation[0][sorted(list(set(metricList + [metric]))).index(metric)]
    metric_test = scorePopulation[1][sorted(list(set(metricList + [metric]))).index(metric)]
    print(f"{metric} train = {metric_train},  {metric} test = {metric_test}")
    
    print()
    num_hyper = 0        
    for i in hyperparameter:
        print(f"{i} = {population[num_hyper]}")
        num_hyper += 1