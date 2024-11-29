from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from graphics import distanciaArremessos, acertosErros, arremessoQuarto
import os

app = Flask(__name__)

# Função para carregar e processar os dados
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data['shot_made_flag'] = data['EVENT_TYPE'].apply(
        lambda x: 1 if x == 'Made Shot' else 0)
    data.rename(columns={'SHOT_DISTANCE(FT)': 'shot_distance'}, inplace=True)
    return data

# Função para treinar o modelo
def modeloTreinamento(data):
    X = data[['shot_distance', 'MINS_LEFT', 'SECS_LEFT', 'QUARTER']]
    y = data['shot_made_flag']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Função para fazer previsões
def sucessoArremesso(model, shot_distance, mins_left, secs_left, quarter):
    prediction = model.predict(
        [[shot_distance, mins_left, secs_left, quarter]])[0]
    probability = model.predict_proba(
        [[shot_distance, mins_left, secs_left, quarter]])[0][1]
    return prediction, probability

# Rotas
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    data = load_and_clean_data('data/NBA_2024_Shots.csv')
    model = modeloTreinamento(data)
    shot_distance = float(request.form['shot_distance'])
    mins_left = int(request.form['mins_left'])
    secs_left = int(request.form['secs_left'])
    quarter = int(request.form['quarter'])
    prediction, probability = sucessoArremesso(
        model, shot_distance, mins_left, secs_left, quarter)
    return render_template('result.html', prediction=prediction, probability=probability)


@app.route('/visualize')
def visualize():
    data = load_and_clean_data('data/NBA_2024_Shots.csv')
    distanciaArremessos(data)
    acertosErros(data)
    arremessoQuarto(data)

    return render_template('visualize.html')


if __name__ == '__main__':
    app.run(debug=True)
