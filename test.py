import pandas as pd
import numpy as np

df1 = pd.read_csv("voting_classifier.csv")
df2 = pd.read_csv("weighted_average.csv")

#df1['new column that will contain the comparison results'] = \
#    np.where(condition,'value if true','value if false')

df1['pricesMatch?'] = np.where(df1['Ensemble_prediction'] == df2['Ensemble_prediction'], 'True', 'False')
print(df1)
count = df1[df1["pricesMatch?"]==False]["pricesMatch?"].sum()
print(count)