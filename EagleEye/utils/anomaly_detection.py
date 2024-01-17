import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib  

class IsolationForestTrainer:
    def __init__(self):
        self.model = None
        self.features = ['CustomerID', 'AccountBalance', 'LastLoginDays', 'Age', 'TransactionID', 'Amount']

    def train_and_save_model(self, data_path="static/data/customer_info.csv", model_path="static/anomaly_detection/model.joblib"):
        customer_info = pd.read_csv(data_path)
        customer_info_clean = customer_info.dropna()

        customer_info_clean['LastLogin'] = pd.to_datetime(customer_info_clean['LastLogin']).copy()
        customer_info_clean['LastLoginDays'] = (customer_info_clean['LastLogin'] - customer_info_clean['LastLogin'].min()).dt.days
        customer_info_train = customer_info_clean[self.features]

        self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01)
        self.model.fit(customer_info_train)

        joblib.dump(self.model, model_path)
        print("Model trained and saved successfully.")

class IsolationForestInference:
    model = None  
    features = ['CustomerID', 'AccountBalance', 'LastLoginDays', 'Age', 'TransactionID', 'Amount']

    @staticmethod
    def load_model(model_path):
        IsolationForestInference.model = joblib.load(model_path)
        print("Model loaded successfully.")

    def preprocess_input(cls, input_data):
        input_df = pd.DataFrame([input_data], columns=cls.features)
        
        input_df['LastLoginDays'] = (input_df['LastLoginDays'] - input_df['LastLoginDays'].min()).dt.days
        return input_df

    def inference_single_input(cls, input_data):
        if cls.model is None:
            raise ValueError("Model not loaded. Load the model using load_model method.")

        input_df = cls.preprocess_input(input_data)

        prediction = cls.model.predict(input_df)

        return prediction

# Example usage:
# Training and saving the model
# trainer = IsolationForestTrainer()
# trainer.train_and_save_model("customer_info.csv", "isolation_forest_model.joblib")

# # Loading the model and making inference
# InferenceModel.load_model("isolation_forest_model.joblib")

# # Inference for a single input
# input_data = {'CustomerID': 123, 'AccountBalance': 500, 'LastLoginDays': 30, 'Age': 25, 'TransactionID': 987, 'Amount': 100}
# result = InferenceModel.inference_single_input(input_data)
# print("Inference Result:", result)
