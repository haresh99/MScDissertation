import numpy as np
import pandas as pd
from os import listdir

# Establish variables
files = listdir("./")
tables = []
predictions = []
probs = []
voting_classifier_pred = []
threshold = .5

# Save necessary files in tables list
for filename in files:
    if "." in filename:
        if filename.split('.')[1] == 'csv':
            tables.append(pd.read_csv(filename))

# Order files by col1 and col2
for table_id in range(len(tables)):
    tables[table_id] = tables[table_id].sort_values(by=['Info_UID', 'Info_center_pos'])
    # Save predictions in predictions list and probabilities in probs list
    predictions.append(tables[table_id].iloc[:, -1].values)
    probs.append(tables[table_id].iloc[:, -2].values)

# Transpose matrix
results = np.transpose(np.stack(tuple(predictions)))
probs = np.transpose(np.stack(tuple(probs)))

# Vote for ensemble prediction (hard voting and soft voting)
for idx, r in enumerate(results):
    sum = r.sum()

    if sum < 0:
        voting_classifier_pred.append(-1)
    elif sum == 0:
        avg_prob = probs[idx].sum()/len(probs[idx])
        if avg_prob >= threshold:
            voting_classifier_pred.append(1)
        else:
            voting_classifier_pred.append(-1)
    else:
        voting_classifier_pred.append(1)

final_table = tables[0].iloc[:, 0:2]
#final_table['VotingClassifier_prediction'] = voting_classifier_pred
final_table['VC_Ensemble_prediction'] = voting_classifier_pred

final_table.to_csv('voting_classifier.csv', index=False)