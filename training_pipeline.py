import config
from sklearn import preprocessing
import pickle
from example_models import models
import os



def train_pipeline(kfold, model, train_df, test_df):
    X_train = train_df[train_df["kfold"] != 0].reset_index(drop=True)
    y_train = X_train.isFraud.values

    X_val = train_df[train_df["kfold"] == 0].reset_index(drop=True)
    y_val = X_val.isFraud.values

    X_train = X_train.drop(["isFraud", "kfold"], axis=1)
    X_val = X_val.drop(["isFraud", "kfold"], axis=1)

    label_encoders = {}
    for feature in config.CATEGORICAL_FEATURES:
        label = preprocessing.LabelEncoder()

        X_train.loc[:, feature] = X_train.loc[:, feature].astype(str).fillna("NONE")
        X_val.loc[:, feature] = X_val.loc[:, feature].astype(str).fillna("NONE")
        test_df.loc[:, feature] = test_df.loc[:, feature].astype(str).fillna("NONE")

        label.fit(X_train[feature].values.tolist() + 
                X_val[feature].values.tolist() + 
                test_df[feature].values.tolist())
        
        X_train.loc[:, feature] = label.transform(X_train[feature].values.tolist())
        X_val.loc[:, feature] = label.transform(X_val[feature].values.tolist())
        test_df.loc[:, feature] = label.transform(test_df[feature].values.tolist())
        label_encoders[feature] = label
        
    test_df.to_csv(config.DATA_DIR + "test_df.csv", index=False)
    clf = models[model]
    clf.fit(X_train.fillna(0), y_train)
    preds = clf.predict_proba(X_val.fillna(0))[:, 1]

    save_path = config.MODEL_OUTPUT + '/' + model
    if not os.path.exists(save_path):
        # If it doesn't exist, create it
        os.makedirs(save_path)

    save_path_model = save_path + '/trained_model_fold_' + str(kfold) + ".pkl"
    save_path_preds = save_path + '/pred_fold_' + str(kfold) 
    save_path_y_val = save_path + '/y_val_' + str(kfold)

    pickle.dump(clf, open(save_path_model, 'wb'))
    pickle.dump(preds, open(save_path_preds, 'wb'))
    pickle.dump(y_val, open(save_path_y_val, 'wb'))

    print("training finished")
