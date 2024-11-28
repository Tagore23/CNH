import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Dados fornecidos
csat_data = {
    'capacidade_operacional': [6, 6, 5, 4, 6, 6, 8, 5, 3, 5, 5, 6, 5, 1, 5, 8, 7, 0, 8, 5, 8, 3, 10, 9, 0, 3, 7, 10, 5, 10, 6, 8, 5, 5],
    'adequacao_operacao': [6, 5, 5, 3, 6, 5, 8, 8, 2, 6, 7, 5, 4, 7, 5, 8, 7, 0, 7, 5, 8, 5, 10, 8, 0, 3, 7, 10, 5, 10, 6, 8, 5, 8],
    'facilidade_operacao': [6, 7, 3, 6, 6, 6, 10, 8, 3, 6, 8, 5, 3, 8, 5, 8, 7, 0, 9, 5, 8, 7, 7, 7, 0, 3, 7, 7, 5, 4, 8, 8, 8, 10],
    'conforto_ergonomia': [6, 7, 3, 3, 6, 6, 10, 8, 3, 2, 8, 6, 6, 1, 8, 8, 8, 0, 9, 5],
    'disponibilidade_confiabilidade': [1, 5, 3, 6, 5, 1, 8, 5, 3, 1, 3, 4, 2, 0, 5, 6, 6, 0, 5, 4, 0, 0, 5, 7, 0, 0, 2, 7, 0, 4, 6, 1, 2, 0],
    'facilidade_manutencao': [1, 7, 3, 6, 7, 6, 8, 6, 2, 5, 7, 5, 5, 8, 5, 8, 7, 0, 5, 8, 8, 5, 4, 3, 0, 0, 6, 7, 5, 9, 8, 8, 6, 8],
    'custo_manutencao': [1, 5, 2, 6, 5, 2, 6, 3, 4, 5, 4, 5, 0, 4, 3, 0, 4, 2, 7, 5, 3, 5, 0, 1, 5, 5, 0, 5, 8, 1, 9, 5],
    'consumo_combustivel': [6, 5, 3, 5, 4, 2, 10, 3, 2, 3, 7, 4, 5, 5, 4, 6, 3, 0, 9, 6, 7, 3, 7, 7, 0, 4, 6, 6, 5, 9, 8, 7, 5, 8],
    'adaptabilidade': [6, 5, 5, 7, 7, 6, 10, 8, 3, 3, 7, 4, 5, 8, 0, 8, 6, 0, 9, 6, 8, 5, 9, 8, 0, 2, 7, 0, 9, 8, 8, 5],
    'uso_piloto_automatico': [8, 7, 2, 7, 7, 7, 10, 8, 1, 6, 8, 5, 2, 1, 2, 8, 8, 0, 8, 0, 8, 5, 4, 8, 0, 0, 7, 8, 5, 9, 6, 6],
    'geracao_transmissao_dados_frota': [0, 7, 3, 6, 6, 6, 10, 2, 1, 8, 6, 2, 0, 8, 8, 8, 5, 4, 0, 0, 8, 8, 0, 6, 7, 8, 9],
    'geracao_transmissao_dados_agricola': [2, 8, 4, 4, 6, 6, 10, 2, 1, 8, 6, 5, 0, 8, 8, 8, 5, 4, 5, 8, 8, 6, 7, 3, 9]
}

# Ajustar o comprimento das listas
max_length = min(len(value) for value in csat_data.values())  # Tamanho da menor lista

# Ajuste as listas para ter o mesmo comprimento
csat_data_adjusted = {key: value[:max_length] for key, value in csat_data.items()}

# Agora podemos criar o DataFrame
df = pd.DataFrame(csat_data_adjusted)

# Criação de uma variável "classe" (exemplo)
df['classe'] = (df.mean(axis=1) > 5).astype(int)  # 1 se a média > 5, caso contrário 0

# Divisão dos dados em treino e teste
X = df.drop('classe', axis=1)
y = df['classe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Gráfico de Importância das Variáveis
feature_importance = rf.feature_importances_
indices = np.argsort(feature_importance)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title('Importância das Variáveis - Random Forest')
plt.barh(range(X.shape[1]), feature_importance[indices], align="center")
plt.yticks(range(X.shape[1]), [features[i] for i in indices])
plt.xlabel('Importância')
plt.show()
