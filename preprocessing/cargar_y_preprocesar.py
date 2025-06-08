import pandas as pd

def preparar_datos(path_data):
    df = pd.read_csv(path_data)
    df['channel_name'] = df['channel_name'].astype("category")

    # Calcular umbral de pico
    df['umbral_pico'] = df.groupby('channel_name')['concurrent_view_count'].transform(lambda x: x.quantile(0.75))
    df['es_pico'] = df['concurrent_view_count'] >= df['umbral_pico']

    # Agrupar canales menos frecuentes
    top_canales = df['channel_name'].value_counts().nlargest(10).index
    if 'otros' not in df['channel_name'].cat.categories:
        df['channel_name'] = df['channel_name'].cat.add_categories(['otros'])
    df['channel_name_mod'] = df['channel_name'].where(df['channel_name'].isin(top_canales), 'otros')

    # Balancear clases
    pico_df = df[df['es_pico']]
    no_pico_df = df[~df['es_pico']].sample(n=len(pico_df), random_state=40)
    df_balanceado = pd.concat([pico_df, no_pico_df])

    # Features y etiquetas
    X = df_balanceado[['hour', 'weekday', 'channel_type', 'channel_name_mod']]
    y = df_balanceado['es_pico']
    X = pd.get_dummies(X)
    X = X.loc[:, X.var() > 0.01]

    return X, y, df_balanceado
