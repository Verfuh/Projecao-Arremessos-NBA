import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de Distribuição da Distância dos Arremessos
def distanciaArremessos(data):
    sns.histplot(data['shot_distance'], bins=20, kde=True)
    plt.title('Distribuição da Distância dos Arremessos')
    plt.savefig('static/distanciaArremessos.png')
    plt.close()

# Gráfico de distribuição de acertos e erros de arremesso
def acertosErros(data):
    sns.countplot(x='shot_made_flag', data=data)
    plt.title('Distribuição de Acertos e Erros')
    plt.xlabel('Acerto (1) / Erro (0)')
    plt.ylabel('Contagem')
    plt.savefig('static/acertosErros.png')
    plt.close()

# Gráfico de boxplot para distância de arremesso por quarto
def arremessoQuarto(data):
    sns.boxplot(x='QUARTER', y='shot_distance', data=data)
    plt.title('Distância dos Arremessos por Quarto')
    plt.xlabel('Quarto')
    plt.ylabel('Distância do Arremesso')
    plt.savefig('static/arremessoQuarto.png')
    plt.close()
