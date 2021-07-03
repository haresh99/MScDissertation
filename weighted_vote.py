# WEIGHTED AVERAGE

import numpy as np
import pandas as pd

# Import files
ABCpred = pd.read_csv('./dev_files/ABCpred_parsed.csv')
LBtope = pd.read_csv('./dev_files/LBtope_parsed.csv')
iBCE_EL = pd.read_csv('./dev_files/iBCE_EL_parsed.csv')
Bepipred2 = pd.read_csv('./dev_files/Bepipred2_parsed.csv')

# Make a list of files
tables = []
results = []
probs = []
threshold = .5

# Set weights (these will be calculated beforehand. (What metric to use? Accuracy? F1 Score?)
# Set the weights in the respective order in which you will scale them [see line 23-26]
weights = [0.75, 0.62, 0.67, 0.58]
weights_sum = sum(weights)

ABCpred_weight = weights[0]/weights_sum  # e.g. 0.75
LBtope_weight = weights[1]/weights_sum  # e.g. 0.62
iBCE_EL_weight = weights[2]/weights_sum  # e.g. 0.67
Bepipred2_weight = weights[3]/weights_sum  # e.g. 0.58

raw_models = {'ABCpred': [ABCpred, ABCpred_weight], 'LBtope': [LBtope, LBtope_weight],
              'iBCE-EL': [iBCE_EL, iBCE_EL_weight], 'Bepipred2': [Bepipred2, Bepipred2_weight]}

# Multiply class values by scaled weights
for table in raw_models.values():
    processed_table = table[0].sort_values(by=['Info_UID', 'Info_center_pos'])
    results.append(processed_table.iloc[:, -1].values * table[1])
    probs.append(processed_table.iloc[:, -2].values)

results = np.transpose(np.stack(tuple(results)))
probs = np.transpose(np.stack(tuple(probs)))

ensemble_pred = []

for idx, r in enumerate(results):
    sum = r.sum()
    if sum < 0:
        ensemble_pred.append(-1)
    elif sum == 0:
        avg_prob = probs[idx].sum()/len(probs[idx])
        if avg_prob >= threshold:
            ensemble_pred.append(1)
        else:
            ensemble_pred.append(-1)
    else:
        ensemble_pred.append(1)

#final_table = raw_models['ABCpred'][0]
final_table = ABCpred.iloc[:, 0:2]
# final_table = raw_models.values()[0][0].iloc[:, 0:2]
final_table['WA_ensemble_prediction'] = ensemble_pred
final_table.to_csv('weighted_average.csv', index=False)
