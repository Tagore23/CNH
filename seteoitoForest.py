import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Dados fornecidos para cada variável
csat_data = {
    'capacidade_operacional': [8, 8, 9, 7, 8, 7, 7, 8, 9, 8, 9, 8, 8, 9, 7, 8, 7, 8, 9, 8, 10, 9, 8, 9, 8, 7, 8, 9, 9, 8, 10, 8, 9, 8, 8, 8, 9, 8, 9, 9],
    'adequacao_operacao': [8, 8, 9, 8, 7, 8, 8, 7, 9, 9, 9, 8, 6, 9, 7, 10, 10, 8, 7, 8, 8, 8, 10, 9, 9, 9, 8, 9, 8, 10, 9, 8, 10, 10, 9, 9, 9, 8, 8, 9],
    'facilidade_operacao': [9, 8, 8, 8, 8, 8, 8, 7, 7, 9, 8, 7, 7, 7, 7, 8, 8, 8, 9, 8, 8, 8, 8, 7, 9, 8, 9, 8, 7, 10, 9, 8, 8, 9, 9, 8, 10, 8, 8, 9],
    'conforto_ergonomia': [9, 8, 9, 8, 8, 8, 10, 9, 8, 9, 8, 9, 10, 9, 10, 8, 8, 8, 9, 9, 8, 9, 9, 10, 9, 8, 10, 9, 9, 8, 10, 10, 9, 10, 8, 10, 10, 9, 9, 10],
    'disponibilidade_confiabilidade': [8, 8, 7, 5, 7, 7, 6, 7, 7, 9, 8, 7, 7, 6, 7, 9, 10, 8, 7, 6, 6, 8, 8, 7, 7, 7, 8, 6, 6, 4, 7, 8, 8, 8, 7, 8, 7, 7, 9, 10],
    'facilidade_manutencao': [8, 8, 6, 6, 8, 7, 7, 7, 8, 8, 9, 7, 8, 7, 8, 2, 8, 7, 8, 10, 7, 9, 9, 8, 8, 9, 8, 8, 8, 4, 7, 7, 7, 9, 10, 8, 7, 10, 10, 10],
    'custo_manutencao': [7, 8, 2, 3, 7, 6, 1, 7, 7, 7, 6, 5, 5, 7, 7, 5, 8, 8, 7, 8, 8, 8, 6, 6, 2, 8, 7, 8, 8, 9, 6, 10, 8, 10, 8, 7, 2, 7, 8, 7],
    'consumo_combustivel': [6, 8, 8, 7, 5, 6, 7, 7, 8, 7, 7, 7, 6, 8, 5, 8, 7, 8, 6, 8, 10, 8, 8, 6, 8, 8, 8, 5, 8, 7, 7, 7, 9, 9, 8, 7, 8, 9, 9, 10],
    'adaptabilidade': [8, 9, 8, 8, 6, 7, 8, 7, 8, 8, 9, 7, 9, 8, 7, 9, 10, 9, 7, 8, 8, 9, 8, 9, 8, 8, 9, 5, 7, 8, 8, 8, 9, 9, 10, 7, 8, 10, 10, 10],
    'uso_piloto_automatico': [9, 8, 8, 8, 7, 7, 8, 7, 6, 9, 7, 7, 4, 7, 5, 10, 6, 8, 10, 10, 7, 9, 9, 9, 8, 8, 8, 8, 6, 8, 8, 8, 9, 8, 10, 7, 8, 10, 10, 10],
    'geracao_transmissao_frota': [8, 8, 0, 8, 5, 7, 6, 7, 8, 7, 9, 3, 5, 6, 5, 8, 10, 10, 8, 8, 8, 7, 8, 9, 10, 9, 9, 10, 8, 10, 9, 10, 10, 5, 10],
    'geracao_transmissao_agricola': [8, 8, 0, 8, 6, 7, 6, 7, 7, 8, 9, 3, 5, 6, 5, 9, 10, 8, 8, 9, 7, 8, 8, 10, 6, 10, 9, 9, 9, 10, 10, 10, 9]
}

# Ajustar o comprimento das listas
max_length = min(len(value) for value in csat_data.values())  # Tamanho da menor lista

# Ajuste as listas para ter o mesmo comprimento
csat_data_adjusted = {key: value[:max_length] for key, value in csat_data.items()}

# Criando o DataFrame
df = pd.DataFrame(csat_data_adjusted)

# Criação de uma variável "classe" (exemplo, 1 para valores maiores que 7, 0 para valores menores ou iguais a 7)
df['classe'] = (df.mean(axis=1) > 7).astype(int)  # 1 se a média > 7, caso contrário 0

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
