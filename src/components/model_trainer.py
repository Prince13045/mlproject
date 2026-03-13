import os 
import sys
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import Custom_Exception
from src.logger import logging
from src.utlis import save_object,evaluate_model


@dataclass
class modeltrainingconfig:
    train_model_file_path=os.path.join("artifacts","model.pkl")

class modeltrainer:
    def __init__(self):
        self.model_trainer_config=modeltrainingconfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("spliting traing and test data ")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "random forest":RandomForestRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Gradient boosting":GradientBoostingRegressor(),
                "Linear regression":LinearRegression(),
                "K-neighbour reggression":KNeighborsRegressor(),
                "XGB regresser":XGBRegressor(),
                "Cat boost regression":CatBoostRegressor(),
                "Ada boost regression":AdaBoostRegressor()
            }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise Custom_Exception("no best model found")
            logging.info(f" best foound model on ttraining and testing data")
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise Custom_Exception(e,sys)
            