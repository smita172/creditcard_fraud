import os
import pickle
import pandas as pd

from creditcard_fraud import settings

model_path = os.path.join(settings.ML_MODELS_PATH, 'xgb_smote_tomek_model.pkl')
# Load the pre-trained model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the predicted dataset
predictions_df = pd.read_csv('C:/Users/Smita/PycharmProjects/creditcard_fraud/static/XGBoost_SMOTE_predictions.csv')
