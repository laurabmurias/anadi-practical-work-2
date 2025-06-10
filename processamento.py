import pandas as pd

western_europe = ['Austria', 'Belgium', 'France', 'Germany', 'Netherlands', 'Switzerland']
eastern_europe = ['Poland', 'Czechia', 'Hungary']
southern_europe = ['Greece', 'Spain', 'Italy', 'Portugal']
northern_europe = ['Sweden', 'Denmark', 'Northern Europe','Finland']

df = pd.read_csv('../AIRPOL_data.csv', sep=';')

categorical_cols = ['Country', 'NUTS_Code', 'Air_Pollutant', 'Outcome']
numeric_cols = ['Affected_Population', 'Populated_Area[km2]', 'Air_Pollution_Average[ug/m3]']

# Função para atribuir região
def assign_region(country):
    if country in western_europe:
        return 'Western Europe'
    elif country in eastern_europe:
        return 'Eastern Europe'
    elif country in southern_europe:
        return 'Southern Europe'
    elif country in northern_europe:
        return 'Northern Europe'
    else:
        return 'Other'

# Função de pré-processamento
def preprocess_dataframe(df):
    df.dropna(axis=1, how='all', inplace=True)
    
    for col in numeric_cols:
        df[col] = df[col].str.replace(',', '.').astype(float)

    df.rename(columns={'Value': 'Premature_Deaths'}, inplace=True)
    df['Premature_Deaths'] = df['Premature_Deaths'].str.replace(',', '.').astype(float)

    df['Region'] = df['Country'].apply(assign_region)
    
    return df

df = preprocess_dataframe(df)
