# Solução de Case


De acordo com o objetivo dado (realizar previsão de vendas de cada produto na base de dados), duas soluções foram propostas: Solução Principal foi construída com base nos produtos que possuem ciclos completos (329produtos), porém foi aplicado algoritmo de KMeans para representação desses produtos em 3 grupos distintos com o objetivo de conseguir detalhar mais o processo de criação dos modelos. Solução Alternativa, também utilizando os 329 produtos, porém com o objetivo de desenvolver uma solução mais simples e com a previsão individual de cada produto.

- Solução Principal contém os seguintes arquivos: 
  - 1-EDA_FeatureEngineering.ipynb: Notebook utilizado para realizar análise exploratória de dados e o processo de feature engineering nos dados.
  - 2-DesenvolvimentoDosModelos.ipynb: Avaliação de performance de modelos.
  - [best_model_C0.pkl, best_model_C1.pkl, best_model_C2.pkl]: Melhor modelo selecionado para realizar previsões em cada um dos grupos.
  - [df_centroides.csv, df_grupo.csv]: DataFrames criados para facilitar desenvolvimento da solução.
  - ExecutandoPrevisaoBaseTest.py: SCRIPT UTILIZADO PARA REALIZAÇÃO DA PREVISÃO DA BASE DE TESTE (CICLOS 202016,202017,202101). Para executá-lo, via prompt, basta mudar o diretório para onde se encontra o arquivo, digite 'ExecutandoPrevisaoBaseTest.py' e enter. O script se encarregará de ler os dados de test, fazer todo processamento de dados e criará 3 arquivos [PrevisaoBaseTest_Grupo0.xlsx, PrevisaoBaseTest_Grupo1.xlsx, PrevisaoBaseTest_Grupo2.xlsx], referentes à previsão das amostras de teste de cada grupo.
  - requirements.txt: Versão das bibliotecas utilizadas
  - utils.py: funções utilizadas para desenvolvimento da solução
  
- Solução Alternativa contém os seguintes arquivos: 
  - Previsao329Produtos.ipynb: Notebook onde é realizado a feature engineering para essa abordagem, a modelagem e a previsão das amostras de test (df_test_produtos_com_ciclos_completos).
