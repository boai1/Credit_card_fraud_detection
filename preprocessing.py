import numpy as np
import pandas as pd
import config
from sklearn.model_selection import StratifiedKFold

def read_data():
    # read the .csv files
    train_identity = pd.read_csv(config.TRAIN_IDENTITY)
    train_transaction = pd.read_csv(config.TRAIN_TRANSACTIONS)

    test_identity = pd.read_csv(config.TEST_IDENTITY)
    test_transaction = pd.read_csv(config.TEST_TRANSACTIONS)

    return train_identity, train_transaction, test_identity, test_transaction

def merge_data(df_transaction, df_identity):
    # merge on TransactionID
    df_merged = df_transaction.merge(df_identity, how="left", on="TransactionID")

    return df_merged


if __name__ == "__main__": 
    train_transactions, train_identity, test_transactions, test_identity = read_data() 
    merged_test = merge_data(test_transactions, test_identity)
    merged_train = merge_data(train_transactions, train_identity)

    del train_transactions, train_identity, test_transactions, test_identity

    for col in merged_test.columns: 
        if "id" in col:
            merged_test.rename(columns={col : col.replace("-", "_")}, inplace=True)
    
    merged_test.to_csv(config.DATA_DIR + "test_df.csv", index=False)

    # create k-folds for Cross Validation strategy
    merged_train["kfold"] = -1 
    
    merged_train = merged_train.sample(frac=1, random_state=0).reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=merged_train, y=merged_train.isFraud.values)):
        print(len(train_idx), len(val_idx))
        merged_train.loc[val_idx, 'kfold'] = fold

    merged_train.to_csv(config.DATA_DIR + "train_df.csv", index=False)
    