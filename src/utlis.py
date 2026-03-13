import os
import sys 
import numpy as np
import pandas as pd
import dill
from src.exception import Custom_Exception
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
         raise Custom_Exception(e,sys)
    


def evaluate_model(x_train,y_train,x_test,y_test,models):
    from sklearn.metrics import r2_score

def evaluate_model(x_train, y_train, x_test, y_test, models):

    try:
        report = {}

        for name, model in models.items():

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score

            return report 


    except:
        pass