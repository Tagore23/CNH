import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dados fornecidos
dados = {
    'Capacidade operacional (hectares por hora) (CSAT)': [6, 6, 5, 4, 6, 6, 8, 5, 3, 5, 5, 6, 5, 1, 5, 8, 7, 0, 8, 5, 8, 3, 10, 9, 0, 3, 7, 10, 5, 10, 6, 8, 5, 5],
    'Adequação às Diversas Operações e Implementos (CSAT)': [6, 5, 5, 3, 6, 5, 8, 8, 2, 6, 7, 5, 4, 7, 5, 8, 7, 0, 7, 5, 8, 5, 10, 8, 0, 3, 7, 10, 5, 10, 6, 8, 5, 8],
    'Facilidade de Operação (CSAT)': [6, 7, 3, 6, 6, 6, 10, 8, 3, 6, 8, 5, 3, 8, 5, 8, 7, 0, 9, 5, 8, 7, 7, 7, 0, 3, 7, 7, 5, 4, 8, 8, 8, 10],
    'Conforto e Ergonomia (CSAT)': [6, 7, 3, 3, 6, 6, 10, 8, 3, 2, 8, 6, 6, 1, 8, 8, 8, 0, 9, 5],
    'Disponibilidade e Confiabilidade Mecânica (CSAT)': [1, 5, 3, 6, 5, 1, 8, 5, 3, 1, 3, 4, 2, 0, 5, 6, 6, 0, 5, 4, 0, 0, 5, 7, 0, 0, 2, 7, 0, 4, 6, 1, 2, 0],
    'Facilidade para Realização de Manutenções (CSAT)': [1, 7, 3, 6, 7, 6, 8, 6, 2, 5, 7, 5, 5, 8, 5, 8, 7, 0, 5, 8, 8, 5, 4, 3, 0, 0, 6, 7, 5, 9, 8, 8, 6, 8],
    'Custo de Manutenção (CSAT)': [1, 5, 2, 6, 5, 2, 6, 3, 4, 5, 4, 5, 0, 4, 3, 0, 4, 2, 7, 5, 3, 5, 0, 1, 5, 5, 0, 5, 8, 1, 9, 5],
    'Consumo de Combustível (Litros por Hectare) (CSAT)': [6, 5, 3, 5, 4, 2, 10, 3, 2, 3, 7, 4, 5, 5, 4, 6, 3, 0, 9, 6, 7, 3, 7, 7, 0, 4, 6, 6, 5, 9, 8, 7, 5, 8],
    'Adaptabilidade às Mais Diversas Condições de Trabalho (CSAT)': [6, 5, 5, 7, 7, 6, 10, 8, 3, 3, 7, 4, 5, 8, 0, 8, 6, 0, 9, 6, 8, 5, 9, 8, 0, 2, 7, 0, 9, 8, 8, 5, 8],
    'Facilidade de Uso do Piloto Automático (CSAT)': [8, 7, 2, 7, 7, 7, 10, 8, 1, 6, 8, 5, 2, 1, 2, 8, 8, 0, 8, 0, 8, 5, 4, 8, 0, 0, 7, 8, 5, 9, 6, 6, 7, 10],
    'Geração e Transmissão de Dados para Gestão da Frota (CSAT)': [0, 7, 3, 6, 6, 6, 10, 2, 1, 8, 6, 2, 0, 8, 8, 8, 5, 4, 0, 0, 8, 8, 0, 6, 7, 8, 9],
    'Geração e Transmissão de Dados para Gestão Agrícola (CSAT)': [2, 8, 4, 4, 6, 6, 10, 2, 1, 8, 6, 5, 0, 8, 8, 8, 5, 4, 5, 8, 8, 6, 7, 3, 9]
}

# Função para realizar o ajuste polinomial e gerar gráficos separados para moda e regressão
def plot_mode_and_regression(dados):
    for categoria, valores in dados.items():
        # Contar a frequência de cada valor
        contador = Counter(valores)

        # Identificar a moda (valor mais frequente)
        moda = contador.most_common(1)[0][0]

        # Exibir a moda para cada categoria
        print(f"A moda para {categoria} é: {moda}")

        # Gráfico 1: Frequência de cada valor (Moda)
        plt.figure(figsize=(12, 6))
        plt.bar(contador.keys(), contador.values(), color='skyblue', alpha=0.7, label="Frequência dos Valores")
        plt.title(f"Frequência dos valores - {categoria}")
        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.xticks(list(contador.keys()))
        plt.legend()
        plt.show()

        # Gráfico 2: Regressão Polinomial Não Linear
        # Criar um modelo de regressão polinomial de grau 2 (por exemplo)
        x = np.array(list(contador.keys())).reshape(-1, 1)  # Convertendo as chaves para um formato adequado para o modelo
        y = np.array(list(contador.values()))  # Frequências

        # Ajustar um modelo polinomial de 2º grau (quadrático)
        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x)
        model = LinearRegression()
        model.fit(x_poly, y)

        # Gerar valores para a linha de regressão
        x_line = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
        x_line_poly = poly.transform(x_line)
        y_line = model.predict(x_line_poly)

        # Gráfico 2: Plotar a linha de regressão
        plt.figure(figsize=(12, 6))
        plt.scatter(x, y, color='blue', label='Dados reais')
        plt.plot(x_line, y_line, color='red', label="Regressão Polinomial", linewidth=2)
        plt.title(f"Regressão Polinomial - {categoria}")
        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.legend()
        plt.show()

# Gerar os gráficos e mostrar a moda com a regressão não linear para cada categoria
plot_mode_and_regression(dados)