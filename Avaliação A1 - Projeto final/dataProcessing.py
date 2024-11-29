import pandas as pd


def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)

    # Criar a coluna de sucesso do arremesso
    data['shot_made_flag'] = data['EVENT_TYPE'].apply(
        lambda x: 1 if x == "Made Shot" else 0)

    print(data['shot_made_flag'].value_counts())
    data.rename(columns={'SHOT_DISTANCE(FT)': 'shot_distance'}, inplace=True)
    data = data[['shot_distance', 'shot_made_flag', 'BASIC_ZONE',
                 'ZONE_NAME', 'MINS_LEFT', 'SECS_LEFT', 'QUARTER']]

    return data
