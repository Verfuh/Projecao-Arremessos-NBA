from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def modeloTreinamento(data):
    print(data['shot_made_flag'].value_counts())

    X = data[['shot_distance', 'MINS_LEFT', 'SECS_LEFT', 'QUARTER']]
    y = data['shot_made_flag']

    # Verifique se há dados suficientes para ambas as classes
    if y.nunique() == 1:
        raise ValueError(
            "Os dados devem conter exemplos de ambas as classes (Made Shot e Missing Shot).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"Distribuição de y_train: {y_train.value_counts()}")

    # Treino do modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def sucessoArremesso(model, shot_distance, mins_left, secs_left, quarter):
    prediction = model.predict(
        [[shot_distance, mins_left, secs_left, quarter]])[0]
    probability = model.predict_proba(
        [[shot_distance, mins_left, secs_left, quarter]])[0][1]

    return prediction, probability
