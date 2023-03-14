from utils import *
import sqlite3
import pickle
import pandas as pd

if __name__ == "__main__":
    
    con = sqlite3.connect("case_ds_gdem.sqlite3")
    df = pd.read_sql_query("SELECT * from vendas", con)

    con.close()

    df_grupo = pd.read_csv('df_grupo.csv',index_col='Unnamed: 0')
    df_test = df[(df['COD_MATERIAL'].isin(df_grupo.index)) & (df['COD_CICLO'].isin([202016,202017,202101]))].sort_values(by='COD_CICLO')

    df_test = df_test.drop(['QT_VENDA','DES_CATEGORIA_MATERIAL','DES_MARCA_MATERIAL'],axis=1)

    df_agrupado = agrupando_dados(df_test,df_grupo)


    if (df_agrupado.isnull().sum().sum() != 0):
        for i in df_agrupado.isna().any()[lambda x: x]:
            df_agrupado[i].fillna(df_agrupado[i].mean(),inplace=True)   

    df_agrupado = df_agrupado.drop(['C1_PCT_DESCONTO_min','C1_FLG_CAMPANHA_MKT_A_mean'],axis=1)


    for i in ['C0','C1','C2']:
        if i == 'C0':
            model = pickle.load(open('best_model_C0.pkl', 'rb'))
            df_agrupado[i] = model.predict(df_agrupado.loc[:,df_agrupado.columns.str.contains(i)])
            df_agrupado[['C0_COD_CICLO_ANO','C0_COD_CICLO',i]].to_excel('PrevisaoBaseTest_Grupo0.xlsx')
        elif i == 'C1':
            model = pickle.load(open('best_model_C1.pkl', 'rb'))
            df_agrupado[i] = model.predict(df_agrupado.loc[:,df_agrupado.columns.str.contains(i)])
            df_agrupado[['C1_COD_CICLO_ANO','C1_COD_CICLO',i]].to_excel('PrevisaoBaseTest_Grupo1.xlsx')
        elif i == 'C2':
            model = pickle.load(open('best_model_C2.pkl', 'rb'))
            df_agrupado[i] = model.predict(df_agrupado.loc[:,df_agrupado.columns.str.contains(i)])
            df_agrupado[['C2_COD_CICLO_ANO','C2_COD_CICLO',i]].to_excel('PrevisaoBaseTest_Grupo2.xlsx')