from flask import Flask, render_template, request
import pandas as pd
import pickle
import config


# use the XGB classifier for deployment
model_path = config.MODEL_OUTPUT + 'xgb/trained_model_fold_0.pkl'
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_fraud():
    csvfile = request.files['csvfile']
    csvfile_path = "./uploaded_data/" + csvfile.filename
    csvfile.save(csvfile_path)

    test_df = pd.read_csv(csvfile_path)
    preds = model.predict_proba(test_df.fillna(0))[:, 1]

    save_path_preds = "./uploaded_data/"  + "predictions.csv"

    df_preds = pd.DataFrame()
    df_preds["prediction_probabilities"] = preds
    df_preds.to_csv(save_path_preds, index=False)

    return render_template('index.html', prediction = save_path_preds)

if __name__ == "__main__": 
    app.run(port=3000, debug=True)