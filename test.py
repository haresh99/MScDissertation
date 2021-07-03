# import pandas as pd
# import numpy as np
#
# threshold = 0.5
# table = []
#
#
# probs = {'A': [0.45, 0.62, 0.25, 0.82],
#         'B': [0.70, 0.14, 0.72, 0.54],
#         'C': [0.34, 0.53, 0.63, 0.32],
#         'D': [0.42, 0.63, 0.76, 0.65]
#         }
# df = pd.DataFrame(probs, columns = ['A', 'B', 'C', 'D'])
# df.loc[:, :] = df.loc[:, :].div(df.sum(axis=1), axis=0)
# print (df.to_numpy())
#
# df = df.to_numpy()
#
#
# # print(df)
# # df.to_numpy()
# # print(df)
#
# for idx, p in enumerate(df):    sum = p.sum()
#     # Si la suma < 0.5, class = -1; si la suma > 0.5, class = 1.
#
#     if sum < threshold:
#         table.append(-1)
#     else:
#         table.append(1)

#=====================================

import pandas as pd
import numpy as np

threshold = 0.5
table = []


probs = {'A': [0.45, 0.62, 0.25, 0.82, 0.34],
        'B': [0.70, 0.14, 0.72, 0.54, 0.43],
        'C': [0.34, 0.53, 0.63, 0.32, 0.43],
        'D': [0.42, 0.63, 0.76, 0.65, 0.43]
        }
df = pd.DataFrame(probs, columns = ['A', 'B', 'C', 'D'])
#df.loc[:, :] = df.loc[:, :].div(df.sum(axis=1), axis=0)
print (df.to_numpy())

df = df.to_numpy()


# print(df)
# df.to_numpy()
# print(df)

divisor = df.shape[1]
for idx, p in enumerate(df):
    sum = p.sum()
    # Si la suma < 0.5, class = -1; si la suma > 0.5, class = 1.
    print(sum/divisor)
    if sum/divisor < threshold:
        table.append(-1)
    else:
        table.append(1)


