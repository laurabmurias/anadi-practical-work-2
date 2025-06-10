import pandas as pd

western_europe = ['Austria', 'Belgium', 'France', 'Germany', 'Netherlands', 'Switzerland']
eastern_europe = ['Poland', 'Czechia', 'Hungary']
southern_europe = ['Greece', 'Spain', 'Italy', 'Portugal']
northern_europe = ['Sweden', 'Denmark', 'Northern Europe', 'Finland']

class col:
    COUNTRY = 'Country'
    NUTS = 'NUTS_Code'
    POLLUTANT = 'Air_Pollutant'
    OUTCOME = 'Outcome'
    AFFECTED = 'Affected_Population' # float
    POP_AREA = 'Populated_Area[km2]' # float
    AIR_AVG = 'Air_Pollution_Average[ug/m3]' # float
    DEATHS = 'Permature_Deaths' # float
    REGION = 'Region'
    RESP_DISEASE = 'Respiratory_Disease'

df = pd.read_csv('../AIRPOL_data.csv', sep=';', usecols=[
    col.COUNTRY, col.NUTS, col.POLLUTANT, col.OUTCOME, col.AFFECTED,
    col.POP_AREA, col.AIR_AVG, col.DEATHS
])

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

# Adds Region column to the dataframe
df[col.REGION] = df[col.COUNTRY].apply(assign_region)