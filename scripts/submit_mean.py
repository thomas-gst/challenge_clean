'''
Very basic script to create a prediction based of the channel's mean_log views. Gave an MSLE of 4.94 (private score).
'''


import numpy as np
import pandas as pd
df_test = pd.read_parquet('data/processed/enriched_test.parquet')

df_predictions = pd.DataFrame({
    'ID': df_test['id'],
    'views': df_test['channel_mean_log_views'].apply(lambda x: np.exp(x))
})

df_predictions.to_csv('submissions/mean_submission.csv', index=False)