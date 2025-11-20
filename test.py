import pandas as pd
# ler o dataset artificial_time_series_2/Client01_Server01.csv
df = pd.read_csv('artificial_changepoints/logaritmica_H/Cliente07_Server07.csv')
# printar as colunas do dataframe
print(df.columns)

total_value = df['value_votes'].sum()
print(f'Total value: {total_value}')