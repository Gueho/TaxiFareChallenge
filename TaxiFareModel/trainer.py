# imports
from sklearn import set_config; set_config(display='diagram')
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
# from sklearn.metrics import make_scorer
# from tpot import TPOTRegressor
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib


MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[DE] BER gueho taxichallenge v2"
# rmse = make_scorer(compute_rmse, greater_is_better=False)

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())])
        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])], remainder="drop")
        #create pipeline with RandomForestRegressor
        self.pipeline = Pipeline([('preproc', preproc_pipe),
                             ('model', RandomForestRegressor(n_estimators= 30, max_depth= 15))])
        # self.pipeline = Pipeline([('preproc', preproc_pipe),
        #                      ('model', TPOTRegressor(generations=4, population_size=20, scoring= rmse))])
        # return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        # self.run()
        y_pred = self.pipeline.predict(self.X_test)
        return compute_rmse(y_pred, self.y_test)
        # self.pipeline.evaluate(self.X_test, self.y_test)

      # 🚨 replace with your country code, city, github_nickname and model name and version

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        self.run()
        joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    # get data
    data = get_data(nrows = 50_000)
    # clean data
    data_clean = clean_data(data)
    # set X and y
    X = data_clean.drop(columns='fare_amount')
    y = data_clean['fare_amount']
    # train
    model = Trainer(X, y)
    model.run()
    # evaluate
    rmse = model.evaluate()
    print(rmse)
    model.save_model()
