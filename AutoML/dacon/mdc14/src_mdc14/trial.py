from os.path import join

import nni
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder, RobustScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss



def change_value(df):
    
    df = df.drop("FLAG_MOBIL", axis = 1)
    df["car"] = df["car"].replace({"N":"car_N", "Y":"car_Y"}) # one-hot encoding시 ohe.category_로 변수이름을 부여할 때 겹치므로 변경
    df["reality"] = df["reality"].replace({"N":"reality_N", "Y":"reality_Y"}) 
#     df["Hired"] = df["DAYS_EMPLOYED"].apply(lambda x: 1 if x < 1 else 0)
#     df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(lambda x: 0 if x >= 0 else x)
    df["occyp_type"] = df["occyp_type"].fillna("unknown")
    df["DAYS_BIRTH"] = df["DAYS_BIRTH"] * -1 # values 양수화
    df["begin_month"] = df["begin_month"] * -1 # values 양수화
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(lambda x: 0 if x >= 0 else x*-1)
    return df

def log_scale(df):
    df["DAYS_EMPLOYED"] = np.log1p(df["DAYS_EMPLOYED"])
    df["income_total"] = np.log1p(df["income_total"])
    return df

def preprocess(X_train, X_valid, X_test):
    X_train_tmp = X_train.copy()
    X_valid_tmp = X_valid.copy()
    X_test_tmp = X_test.copy()
    
    X_train_tmp = X_train_tmp.reset_index(drop = True)
    X_valid_tmp = X_valid_tmp.reset_index(drop = True)
    X_test_tmp = X_test_tmp.reset_index(drop = True)
    
    X_train_tmp = change_value(X_train_tmp)
    X_valid_tmp = change_value(X_valid_tmp)
    X_test_tmp = change_value(X_test_tmp)
    
    
    # num_columns sd scaling
    cat_columns = X_train_tmp.select_dtypes(include="object").columns.to_list()
    num_columns = X_train_tmp.select_dtypes(exclude= "object").columns.to_list()
    
    col_to_log = ["DAYS_EMPLOYED", "income_total"]
    categorized_col = ["work_phone", "phone","email"]
    col_to_sd = list(set(num_columns) - set(col_to_log) - set(categorized_col))
    
    sdscaler = StandardScaler()
    
    X_train_tmp[col_to_sd] = sdscaler.fit_transform(X_train_tmp[col_to_sd])
    X_valid_tmp[col_to_sd] = sdscaler.transform(X_valid_tmp[col_to_sd])
    X_test_tmp[col_to_sd] = sdscaler.transform(X_test_tmp[col_to_sd])
    
    
    X_train_tmp = log_scale(X_train_tmp)
    X_valid_tmp = log_scale(X_valid_tmp) 
    X_test_tmp = log_scale(X_test_tmp)

#   ----------------------------------------------------------------------------------
    # encoding 
    ohe = OneHotEncoder(sparse = False)
    concat = pd.concat([X_train_tmp, X_valid_tmp, X_test_tmp], axis = 0)
    ohe.fit(concat[cat_columns])
    
    print(X_train_tmp.shape)
    ohe_columns = []
    for col in ohe.categories_:
        ohe_columns += col.tolist()
        
    train_ohe = ohe.transform(X_train_tmp[cat_columns])
    valid_ohe = ohe.transform(X_valid_tmp[cat_columns])
    test_ohe = ohe.transform(X_test_tmp[cat_columns])
    print(X_train_tmp.shape)
    
    X_train_tmp = pd.concat([X_train_tmp, pd.DataFrame(train_ohe, columns= ohe_columns)], axis = 1)
    X_valid_tmp = pd.concat([X_valid_tmp, pd.DataFrame(valid_ohe, columns= ohe_columns)], axis = 1)
    X_test_tmp = pd.concat([X_test_tmp, pd.DataFrame(test_ohe, columns= ohe_columns)], axis = 1)
    
    print(X_train_tmp.shape)
    X_train_tmp = X_train_tmp.drop(cat_columns, axis = 1)
    X_valid_tmp = X_valid_tmp.drop(cat_columns, axis = 1)
    X_test_tmp = X_test_tmp.drop(cat_columns, axis = 1)
    
    return X_train_tmp, X_valid_tmp, X_test_tmp
    

    
    

    
    
def make_submission(y_test_pred, val_scores):
    ''' 
        내용 추가
        제출 파일 생성 함수 작성
    '''
    df_result = pd.read_csv(join(ROOT_PATH, 'data', 'Credit_card', "sample_submission.csv"))
    df_result.loc[:,1:4] = y_test_pred 
    df_result.to_csv(join(ROOT_PATH,"src_mdc14","result",'model_loss_{:.4f}.csv'.format(val_scores)),index=False)
    
    pass

ROOT_PATH = '/Users/nayo/python_ML/'





def main(params):
    global num_columns, cat_columns

    train_path = join(ROOT_PATH, 'data', 'Credit_card', 'train.csv')
    test_path  = join(ROOT_PATH, 'data', 'Credit_card', 'test.csv')

    data = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target_column = list(set(data.columns) - set(test.columns))
    target = data[target_column[0]].copy()
    del data[target_column[0]]

    le = LabelEncoder()
    target = le.fit_transform(target)
    
    y_test_pred = np.zeros([test.shape[0], le.classes_.shape[0]])
    logloss_score = []

    n_splits = 5
    skf = StratifiedKFold(n_splits, shuffle= True, random_state= 42)
    
    for i, (trn_idx, val_idx) in enumerate(skf.split(data, target)):
        X_train, y_train = data.iloc[trn_idx,:], target[trn_idx]
        X_valid, y_valid = data.iloc[val_idx,:], target[val_idx]


        X_train, X_valid, X_test = preprocess(X_train, X_valid, test)

        model = LGBMClassifier(n_estimators = params['n_estimators'],
                               max_depth = params['max_depth'],
                               num_leaves = params["num_leaves"],
                               objective= "multiclass",                
                               n_jobs= 7,
                               random_state   = 42)

        model.fit(X_train, y_train,
                  eval_set=[[X_train, y_train], [X_valid, y_valid]],
                  eval_metric='multi_logloss',
                  early_stopping_rounds=100,
                  verbose=100)

        logloss_score.append(log_loss(y_valid, model.predict_proba(X_valid)))
        y_test_pred += model.predict_proba(X_test) / n_splits
    
        print("Fold {0}, train logloss: {1}, valid logloss: {2}".format(i, 
                                                                    log_loss(y_train,
                                                                    model.predict_proba(X_train)),
                                                                    log_loss(y_valid,
                                                                    model.predict_proba(X_valid))))
    cv_loss = np.mean(logloss_score)
    
    print("Cross Validation Score : {:.4f}".format(np.mean(cv_loss)))
    nni.report_final_result(cv_loss)
    print('Final result is %g', cv_loss)
    print('Send final result done.')

    make_submission(y_test_pred, cv_loss)
    

if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)