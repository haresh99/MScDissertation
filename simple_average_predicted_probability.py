import numpy as np
import pandas as pd
from os import listdir

# Establish variables
files = listdir("./org_spec_files/Bacteria/")
tables = []
predictions = []
probs = []
simple_av_pred_prob = []
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

# # Scale every row of the probs table
# probs_tmp = pd.DataFrame(probs)
# probs_tmp.loc[:, :] = probs_tmp.loc[:, :].div(probs_tmp.sum(axis=1), axis=0)
# probs = probs_tmp.to_numpy()

# Vote for ensemble prediction (hard voting and soft voting)
divisor = probs.shape[1]
print(probs)
for idx, p in enumerate(probs):
    sum = p.sum()
    print(sum)
    print(divisor)
    print(sum/divisor)
    print("---")
    # Si la suma < 0.5, class = -1; si la suma > 0.5, class = 1.
    if sum/divisor < threshold:
        simple_av_pred_prob.append(-1)
    else:
        simple_av_pred_prob.append(1)

final_table = tables[0].iloc[:, 0:2]
#final_table['VotingClassifier_prediction'] = simple_av_pred_prob
final_table['simple_av_prob_prediction'] = simple_av_pred_prob

final_table.to_csv('simple_av_pred_prob.csv', index=False)
