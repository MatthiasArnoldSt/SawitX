import tensorflow as tf
import xgboost as xgb

class ModelLoader:
    def __init__(self, lstm_path="app/models/lstm_model.h5", xgb_path="app/models/xgb_model.json"):
        self.lstm_model = tf.keras.models.load_model(lstm_path)
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(xgb_path)

    def predict_lstm(self, X_input):
        return self.lstm_model.predict(X_input)

    def predict_xgb(self, X_input):
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X_input)
        return self.xgb_model.predict(dmatrix)
