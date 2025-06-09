def straitfy_validation_set(df_train_val, df_test, val_size=0.15):
    '''
    Stratify the validation set based on the channel distribution of the test set. 
    If there aren't enough samples in the train/val set for a channel, 
    it will take as many as possible and sampel randomly from the remaining samples.
    '''
    test_channels = df_test['channel_int'].unique()
    filtered_train_val = df_train_val[df_train_val['channel_int'].isin(test_channels)]
    if filtered_train_val.empty:
        raise ValueError("No matching channels found between train/val and test sets.")
    test_channel_proportions = filtered_train_val['channel_int'].value_counts(normalize=True)
    n_total = len(filtered_train_val)
    n_val = int(n_total * val_size)
    val_indices = []

    for channel, proportion in test_channel_proportions.items():
        n_channel_val = int(n_val * proportion)
        channel_indices = filtered_train_val[filtered_train_val['channel_int'] == channel].index
        n_channel_val = min(n_channel_val, len(channel_indices))
        val_indices.extend(channel_indices[:n_channel_val])

    if len(val_indices) < n_val:
        remaining_indices = filtered_train_val.index.difference(val_indices)
        additional_indices = remaining_indices[:n_val - len(val_indices)]
        val_indices.extend(additional_indices)
    df_val = filtered_train_val.loc[val_indices]
    df_train = filtered_train_val.drop(index=val_indices)
    return df_train, df_val