import joblib 
import os
from os.path import join
ROOT_PATH = os.getcwd()

def save_object(dict_obj):

    lst_obj = list(dict_obj.values())
    lst_obj_name = list(dict_obj)
    for i in range(len(lst_obj)):
        joblib.dump(lst_obj[i], lst_obj_name[i])

def load_model():
    global model_0, model_1, model_2, model_3, model_4, model_5
    model_0 = joblib.load(os.path.join(ROOT_PATH, "model/model_0"))
    model_1 = joblib.load(os.path.join(ROOT_PATH, "model/model_1"))
    model_2 = joblib.load(os.path.join(ROOT_PATH, "model/model_2"))
    model_3 = joblib.load(os.path.join(ROOT_PATH, "model/model_3"))
    model_4 = joblib.load(os.path.join(ROOT_PATH, "model/model_4"))
    model_5 = joblib.load(os.path.join(ROOT_PATH, "model/model_5"))

    return model_0, model_1, model_2, model_3, model_4, model_5

def load_scaler():
    global scaler_0, scaler_1, scaler_2, scaler_3, scaler_4, scaler_5
    scaler_0 = joblib.load(os.path.join(ROOT_PATH, "scaler/sd_scaler_0"))
    scaler_1 = joblib.load(os.path.join(ROOT_PATH,"scaler/sd_scaler_1"))
    scaler_2 = joblib.load(os.path.join(ROOT_PATH,"scaler/sd_scaler_2"))
    scaler_3 = joblib.load(os.path.join(ROOT_PATH,"scaler/sd_scaler_3"))
    scaler_4 = joblib.load(os.path.join(ROOT_PATH,"scaler/sd_scaler_4"))
    scaler_5 = joblib.load(os.path.join(ROOT_PATH,"scaler/sd_scaler_5"))

    return scaler_0, scaler_1, scaler_2, scaler_3, scaler_4, scaler_5


if __name__ == "__main__":
    load_model()
    load_scaler()