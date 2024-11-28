import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dados fornecidos (categorias 'CSAT')
dados = {
    'Capacidade operacional (hectares por hora) (CSAT)': [8, 8, 9, 7, 8, 7, 7, 8, 9, 8, 9, 8, 8, 9, 7, 8, 10, 9, 9, 8,
                                                          10, 8, 7, 9, 10, 8, 9, 8, 8, 10, 8, 8, 8, 9, 9, 10, 8, 8, 8,
                                                          9, 9, 10, 10, 9, 7, 10, 9, 10, 10, 10],
    'Adequação às diversas operações e implementos (CSAT)': [8, 8, 9, 8, 7, 8, 8, 7, 9, 9, 9, 8, 6, 9, 7, 10, 10, 8, 7,
                                                             8, 8, 8, 8, 8, 9, 8, 8, 10, 9, 8, 8, 9, 8, 10, 9, 10, 10,
                                                             9, 10, 9, 9, 9, 9, 9, 8, 9, 9, 10, 9, 9],
    'Facilidade de operação (CSAT)': [9, 8, 8, 8, 8, 8, 8, 7, 7, 9, 8, 7, 7, 7, 8, 8, 7, 9, 8, 9, 9, 8, 7, 10, 10, 8, 8,
                                      8, 8, 9, 7, 8, 9, 10, 9, 9, 8, 10, 9, 10, 10, 10, 9, 9, 9, 9, 10, 7, 9, 10],
    'Conforto e Ergonomia (CSAT)': [9, 8, 9, 8, 8, 8, 10, 9, 9, 7, 10, 8, 9, 8, 10, 9, 10, 8, 8, 8, 8, 9, 10, 10, 8, 9,
                                    9, 8, 10, 10, 9, 8, 9, 9, 10, 9, 10, 9, 10, 10, 9, 9, 10, 10, 9, 10],
    'Disponibilidade e Confiabilidade Mecânica (CSAT)': [8, 8, 7, 5, 7, 7, 6, 7, 7, 9, 8, 7, 7, 6, 7, 9, 10, 8, 7, 6, 6,
                                                         8, 8, 7, 7, 8, 8, 8, 6, 6, 4, 7, 8, 8, 8, 7, 8, 7, 7, 9, 10, 8,
                                                         8, 5, 8, 10, 10, 1, 9, 7],
    'Facilidade para Realização de Manutenções (CSAT)': [8, 8, 6, 6, 8, 7, 7, 7, 8, 8, 9, 7, 8, 7, 8, 2, 8, 7, 8, 10, 7,
                                                         9, 9, 8, 8, 9, 8, 8, 8, 4, 7, 7, 7, 9, 10, 8, 7, 10, 10, 10, 8,
                                                         8, 2, 7, 8, 9, 10, 8, 8, 9],
    'Custo de Manutenção (CSAT)': [7, 8, 2, 3, 7, 6, 1, 7, 7, 7, 6, 5, 5, 7, 7, 5, 8, 8, 7, 8, 8, 8, 6, 6, 2, 8, 7, 8,
                                   8, 8, 9, 6, 10, 8, 10, 8, 7, 2, 7, 8, 7, 9, 9, 6, 7, 8, 5, 8, 8, 6],
    'Consumo de Combustível (Litros por Hectare) (CSAT)': [6, 8, 8, 7, 5, 6, 7, 7, 8, 7, 7, 7, 6, 8, 5, 8, 7, 8, 6, 8,
                                                           10, 8, 8, 6, 8, 8, 8, 5, 8, 7, 7, 7, 9, 9, 8, 7, 8, 9, 9, 10,
                                                           8, 7, 7, 7, 10, 7, 7, 8, 9, 10],
    'Adaptabilidade às Mais Diversas Condições de Trabalho (CSAT)': [8, 9, 8, 8, 6, 7, 8, 7, 8, 8, 9, 7, 9, 8, 7, 9, 10,
                                                                     9, 7, 8, 8, 9, 8, 9, 5, 7, 8, 8, 8, 9, 9, 10, 7, 8,
                                                                     10, 10, 10, 8, 8, 10, 8, 9, 10, 9, 10, 10],
    'Facilidade de Uso do Piloto Automático (CSAT)': [9, 8, 8, 8, 7, 7, 8, 7, 6, 9, 7, 7, 4, 7, 5, 10, 6, 8, 10, 10, 7,
                                                      9, 9, 9, 8, 8, 8, 8, 6, 8, 8, 8, 9, 8, 10, 7, 8, 10, 10, 10, 8, 6,
                                                      10, 7, 9, 9, 6, 6, 10, 10],
    'Geração e Transmissão de Dados para Gestão da Frota (CSAT)': [8, 8, 0, 8, 5, 7, 6, 7, 8, 7, 9, 3, 5, 6, 5, 8, 10,
                                                                   10, 8, 8, 8, 7, 8, 9, 10, 8, 9, 10, 6, 0, 10, 10, 9,
                                                                   9, 10, 8, 10, 10, 10, 9, 4, 10, 10, 10, 5, 10, 8],
    'Geração e Transmissão de Dados para Gestão Agrícola (CSAT)': [8, 8, 0, 8, 6, 7, 6, 7, 7, 8, 9, 3, 5, 6, 5, 9, 9,
                                                                   10, 8, 8, 9, 7, 8, 10, 8, 8, 10, 6, 0, 10, 10, 9, 9,
                                                                   9, 10, 10, 10, 9, 9, 10, 10, 10, 5, 10, 8]
}


# Função para gerar o gráfico de moda
def plot_mode(dados):
    for categoria, valores in dados.items():
        contador = Counter(valores)
        moda = contador.most_common(1)[0][0]
        print(f"A moda para {categoria} é: {moda}")

        # Gráfico de barras para moda
        plt.figure(figsize=(10, 6))
        plt.bar(contador.keys(), contador.values(), color='skyblue')
        plt.title(f"Frequência dos valores - {categoria}")
        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.xticks(list(contador.keys()))
        plt.show()


# Função para regressão não linear (2º grau)
def plot_regression(dados):
    for categoria, valores in dados.items():
        X = np.array(list(range(1, len(valores) + 1))).reshape(-1, 1)
        y = np.array(valores).reshape(-1, 1)

        # Criar o modelo de regressão polinomial de grau 2
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        # Gerar valores previstos
        y_pred = model.predict(X_poly)

        # Plotando os dados reais e a linha de regressão
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label="Dados reais")
        plt.plot(X, y_pred, color='red', label="Regressão Polinomial (2º grau)")
        plt.title(f"Regressão Não Linear - {categoria}")
        plt.xlabel("Índice dos Dados")
        plt.ylabel("Valor")
        plt.legend()
        plt.show()


# Gerar gráficos de moda e regressão para cada categoria
plot_mode(dados)
plot_regression(dados)