import optuna
from sklearn.model_selection import TimeSeriesSplit # Divisão dos dados de Treino/Validação no modelo de Séries Temporais
import numpy as np
from lightgbm import LGBMRegressor
import xgboost as xgb # Necessária para utilizar o algoritmo XGBoost
from sklearn.metrics import mean_squared_error
import pandas as pd


aggregations = {
    # Criando duas variáveis (ANO e CICLO) a partir da variável COD_CICLO
    'COD_CICLO': [lambda x: int(str(x.iloc[0])[:4]),lambda x: int(str(x.iloc[0])[4:])],
    # Média da variável FLG_DATA
    'FLG_DATA': 'mean',
    # Média da variável FLG_CAMPANHA_MKT_A
    'FLG_CAMPANHA_MKT_A': 'mean',
    # Média da variável FLG_CAMPANHA_MKT_B
    'FLG_CAMPANHA_MKT_B': 'mean',
    # Média da variável FLG_CAMPANHA_MKT_C
    'FLG_CAMPANHA_MKT_C': 'mean',
    # Média da variável FLG_CAMPANHA_MKT_D
    'FLG_CAMPANHA_MKT_D': 'mean',
    # Média, Mínimo e Máximo da variável PCT_DESCONTO
    'PCT_DESCONTO': ['mean','min','max'],
    # Média, Mínimo e Máximo da variável VL_PRECO
    'VL_PRECO': ['mean','min','max']
}

def agrupando_dados(df_test,df_grupo):
    df_test_agrupado = pd.DataFrame()
    for i in df_grupo.Grupo.unique():
        # Agrupando por Ciclo cada grupo e aplicando as funções mencionadas acima
        df_aux = df_test[df_test['COD_MATERIAL'].isin(df_grupo[df_grupo['Grupo'] == i].index)].groupby('COD_CICLO').agg(aggregations)
        # Transformando o nome das colunas de MultiIndex para String único
        df_aux.columns = [f'C{i}_' + "_".join(x) for x in df_aux.columns.ravel()]
        df_aux = df_aux.reset_index(drop=True)
        # Renomeando variáveis de Ano e Ciclo.
        df_aux = df_aux.rename(columns={f'C{i}_COD_CICLO_<lambda_0>':f'C{i}_COD_CICLO_ANO',
                                                      f'C{i}_COD_CICLO_<lambda_1>':f'C{i}_COD_CICLO'})
        df_test_agrupado = df_test_agrupado.join(df_aux,how='right')
    return df_test_agrupado



def objective_lgbm(trial, X, y):
    # Definindo quais parâmetros será realizado a otimização no espaço de busca
    param_grid = {
       "n_estimators": trial.suggest_int("n_estimators", 100, 1000,step=10),
       "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
       "num_leaves": trial.suggest_int("num_leaves", 2, 30, step=2),
       "max_depth": trial.suggest_int("max_depth", 3, 12),
       "feature_fraction": trial.suggest_float(
           "feature_fraction", 0.2, 0.95, step=0.1
       ),
    }
    
    # Definindo o método de Cross-Validation que será utilizado na otimização dos parâmetros
    cv = TimeSeriesSplit(n_splits=5)

    # RMSE em cada fold
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LGBMRegressor(**param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",
            early_stopping_rounds=10
        )
        
        preds = model.predict(X_test)
        cv_scores[idx] = mean_squared_error(y_test, preds,squared=False)

    return np.mean(cv_scores)



def objective_xgb(trial, X, y):
  # Definindo quais parâmetros será realizado a otimização no espaço de busca

    param_grid = {
       "n_estimators": trial.suggest_int("n_estimators", 100, 1000,step=10),
       "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
       "num_leaves": trial.suggest_int("num_leaves", 2, 30, step=2),
       "max_depth": trial.suggest_int("max_depth", 3, 12),
       "feature_fraction": trial.suggest_float(
           "feature_fraction", 0.2, 0.95, step=0.1
       ),
    }
    
    # Definindo o método de Cross-Validation que será utilizado na otimização dos parâmetros
    cv = TimeSeriesSplit(n_splits=5)

    # RMSE em cada fold
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(**param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",
            early_stopping_rounds=10
        )
        
        preds = model.predict(X_test)
        cv_scores[idx] = mean_squared_error(y_test, preds, squared=False)

    return np.mean(cv_scores)