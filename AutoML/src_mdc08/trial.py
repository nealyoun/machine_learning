from os.path import join

import nni
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from lightgbm import LGBMClassifier


def change_cat_val(df):
    df["religion"] = df["religion"].replace({"Other" : "religion_other"})
    df["race"]= df["race"].replace({"Other" : "race_other"})
    
    return df

def preprocess(X_train, X_valid, X_test):
    X_train_tmp = X_train.copy()
    X_valid_tmp = X_valid.copy()
    X_test_tmp = X_test.copy()
    
    X_train_tmp = X_train_tmp.reset_index(drop = True)
    X_valid_tmp = X_valid_tmp.reset_index(drop = True)
    X_test_tmp = X_test_tmp.reset_index(drop = True)
    
    
    #NA
#     if X_train.isna().sum().sum() != 0:

    

    # scaling
    sdscaler = StandardScaler()
    X_train_tmp[num_columns] = sdscaler.fit_transform(X_train_tmp[num_columns])
    X_valid_tmp[num_columns] = sdscaler.transform(X_valid_tmp[num_columns])
    X_test_tmp[num_columns] = sdscaler.transform(X_test_tmp[num_columns])
    
    
    # encoding
    ohe = OneHotEncoder(sparse= False)
    concat = pd.concat([X_train_tmp, X_valid_tmp, X_test_tmp], axis = 0)
    concat = change_cat_val(concat)
    ohe.fit(concat[cat_columns])
    

    X_train_tmp = change_cat_val(X_train_tmp)
    X_valid_tmp = change_cat_val(X_valid_tmp)
    X_test_tmp = change_cat_val(X_test_tmp)
    
    train_ohe = ohe.transform(X_train_tmp[cat_columns])
    valid_ohe = ohe.transform(X_valid_tmp[cat_columns])
    test_ohe = ohe.transform(X_test_tmp[cat_columns])
    
    ohe_cat = []
    for cat in ohe.categories_:
        ohe_cat += cat.tolist()
    
    
    X_train_tmp = pd.concat([X_train_tmp, pd.DataFrame(train_ohe, columns = ohe_cat)], axis = 1)
    X_valid_tmp = pd.concat([X_valid_tmp, pd.DataFrame(valid_ohe, columns = ohe_cat)], axis = 1)
    X_test_tmp = pd.concat([X_test_tmp, pd.DataFrame(test_ohe, columns = ohe_cat)], axis = 1)
    
    X_train_tmp = X_train_tmp.drop(cat_columns, axis = 1)
    X_valid_tmp = X_valid_tmp.drop(cat_columns, axis = 1)
    X_test_tmp = X_test_tmp.drop(cat_columns, axis = 1)
    return X_train_tmp, X_valid_tmp, X_test_tmp
    
    
    
    
def make_submission(oof_pred, val_scores):
    ''' 
        내용 추가
        제출 파일 생성 함수 작성
    '''
    submit_path = join(ROOT_PATH, "src_mdc08", "result")
    df_result = pd.read_csv(join(ROOT_PATH,"data","MDC08", "sample_submission.csv"))
    df_result["voted"] = oof_pred[:,1]
    df_result.to_csv(join(submit_path,'model_loss_{:.4f}.csv'.format(val_scores)), index=False)
    pass
    
    
ROOT_PATH = '/Users/nayo/python_ML/'

def main(params):
    global num_columns, cat_columns


    data = pd.read_csv(join(ROOT_PATH, "data", "MDC08", "train.csv"))
    test = pd.read_csv(join(ROOT_PATH, "data", "MDC08", "test.csv"))

    target_column = list(set(data.columns) - set(test.columns))
    target = data[target_column[0]].copy()

    del data["index"], test["index"], data[target_column[0]]
                                           
    
    cat_columns = data.select_dtypes(include = "object").columns.to_list()
    num_columns = data.select_dtypes(exclude = "object").columns.to_list()
                                           
                                           
    le = LabelEncoder()
    target = le.fit_transform(target)

                                                 
                                                         
    

    n_splits = 5
    skf = StratifiedKFold(n_splits= n_splits, random_state= 42, shuffle= True)
    lgbm_clf = LGBMClassifier(n_jobs = -1, random_state = 42)
    
                                           
                                           
    val_score = []
    oof_pred = np.zeros([test.shape[0], le.classes_.shape[0]])


    for i, (trn_idx, val_idx) in enumerate(skf.split(data, target)):
        X_train, y_train = data.iloc[trn_idx,:], target[trn_idx]
        X_valid, y_valid = data.iloc[val_idx,:], target[val_idx]


        X_train, X_valid, X_test = preprocess(X_train, X_valid, test)

        lgbm_clf = LGBMClassifier(n_estimators   = params['n_estimators'],
                                      max_depth      = params['max_depth'],
                                      min_data_in_leaf = params['min_data_in_leaf'],
                                      num_leaves = params["num_leaves"],
                                      n_jobs    = 7,
                                      random_state = 42
                                  )

#     "n_estimators": {"_type":"randint", "_value":[100, 150]},

#     "max_depth": {"_type":"randint", "_value":[2, 10]},

#     "min_data_in_leaf": {"_type":"randint", "_value":[100, 1000]},

#     "num_leaves
                                           
                                           
                                           
        lgbm_clf.fit(X_train, y_train, 
                     early_stopping_rounds= 100,
                     eval_set= [[X_train, y_train],[X_valid, y_valid]],
                     eval_metric= "auc", 
                     verbose= 100)
                                                                       

        trn_auc = roc_auc_score(y_train, lgbm_clf.predict_proba(X_train)[:,1])
        val_auc = roc_auc_score(y_valid, lgbm_clf.predict_proba(X_valid)[:,1])

        print("{0} Fold, Train AUC: {1:.4f}, Valid AUC: {2:.4f}".format(i, trn_auc, val_auc))
        print("---"*30)

        val_score.append(val_auc)
        oof_pred += lgbm_clf.predict_proba(X_test) / n_splits
    cv_loss = np.mean(val_score)
    print("Cross Validation Score : {:.4f}".format(cv_loss))                                      
                                           
                                           

    nni.report_final_result(cv_loss)
    print('Final result is %g', cv_loss)
    print('Send final result done.')

    make_submission(oof_pred, cv_loss)

if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)