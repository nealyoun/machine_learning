import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings("ignore")







# 전체 데이터셋 처리 

def drop_raw_df_na(df):
    df_tmp = df.copy()
    df_tmp = df_tmp.drop(df_tmp[df_tmp["senior"].isna()].index).reset_index(drop = True)
    
    return df_tmp


# 평당 가격 변환
def get_perprice(df):
    df_tmp = df.copy()
    df_tmp["perprice"] = df_tmp["price"]/df_tmp["area"]
    df_tmp = df_tmp.drop("price", axis = 1)
    
    return df_tmp

def impute_na(df):
    df_tmp = df.copy()
    
    imp_mean = IterativeImputer(random_state= 42)
    tmp_col = df_tmp.select_dtypes(exclude = ["object", "category"]).columns.to_list()
    df_tmp[tmp_col] = imp_mean.fit_transform(df_tmp[tmp_col])
    
    return df_tmp


# 상위 k% row 제거
def drop_top_price(df, k):
    tmp_df = df.copy()
    top_price = tmp_df.groupby("gu")["price"].quantile(k).to_frame().T
    top_price_col = top_price.columns.to_list()


    for col in top_price_col: 
        gu_idx = tmp_df[tmp_df["gu"] == col]["price"].index
#         tmp_df.loc[gu_idx, "price"] = tmp_df.loc[gu_idx, "price"].apply(lambda x: top_price.loc[:, col][0] if x >= top_price.loc[:, col][0] else x)
        tmp_df.loc[gu_idx, "price"] = tmp_df.loc[gu_idx, "price"].apply(lambda x: np.nan if x >= top_price.loc[:, col][0] else x)
    tmp_df = tmp_df.dropna()
    tmp_df = tmp_df.reset_index(drop = True)
            
    return tmp_df

def get_pop_rate(df):
    tmp_df = df.copy()
    tmp_df['rate_male'] = tmp_df['male_kor'] / tmp_df['pop']
    tmp_df['rate_female'] = tmp_df['female_kor'] / tmp_df['pop']
    tmp_df['rate_male_f'] = tmp_df['male_for'] / tmp_df['pop']
    tmp_df['rate_female_f'] = tmp_df['female_for'] / tmp_df['pop']
    tmp_df['rate_senior'] = tmp_df['senior'] / tmp_df['pop']
    
    return tmp_df


# -----------------------------------------------------------------------------------------------
# 길, 로, 대로
def get_road(x) :
    con = x.split(' ')[0]
    if con[-2:] == '대로' :
        return 1
    elif con[-1:] == '로' :
        return 2
    else :
        return 3

# encoding -----------------------------------------------------------------------------------------------    
    
def categorize(df):
    df_tmp = df.copy()
    
    df_tmp["gu"] = df_tmp["gu"].astype("category")
    df_tmp["doro_trans"] = df_tmp["doro_trans"].astype("category")
#     df_tmp["interest"] = df_tmp["interest"].astype("category")
    df_tmp["floor_level"]  = df_tmp["floor_level"].astype("category")
    df_tmp["tradetype"] = df_tmp["tradetype"].astype("category")
    df_tmp["dong"] = df_tmp["dong"].astype("category")
    
    return df_tmp


# scaling -----------------------------------------------------------------------------------------------

def get_log_scaled(df):
    
    df_tmp = df.copy()
    df_tmp = np.log1p(df_tmp)
    
    return df_tmp


# F.E.-----------------------------------------------------------------------------------------------

def get_year_trans(x):
    year_trans = int(x.split("-")[0])
    
    return year_trans

def get_year_gap(df):
    df_tmp = df.copy()
    df_tmp["year_gap"] = df_tmp["year_trans"] - df_tmp["built"]
    
    return df_tmp


def get_floor_level(x):
    if x < 0:
        return "under_0"
    elif x <= 5:
        return "btw_1_5"
    elif x <= 20:
        return "btw_6_20"
    else:
        return "over_20"
    

# 초기 데이터 학습용 검증용 분할; 
# 주의: 인코딩 작업 외 분포 확인 및 스케일링 작업은 분할 후 확인 
def split_test_train(df, preprocess = bool):
    
    
    global num_col, log_col, scale_col, cat_col
    
    df_tmp = df.copy()
    
    df_tmp = drop_raw_df_na(df_tmp)
    
    if preprocess == True:
        df_tmp = impute_na(df_tmp)
        df_tmp = drop_top_price(df_tmp, 0.80)

        df_tmp = get_perprice(df_tmp)   
        df_tmp["doro_trans"] = df_tmp["doro"].apply(lambda x: get_road(x))
        df_tmp["year_trans"] = df_tmp["date"].apply(lambda x: get_year_trans(x))
        df_tmp = get_year_gap(df_tmp)

    
        df_tmp["floor_level"] = df_tmp["floor"].apply(lambda x: get_floor_level(x))
        df_tmp["floor"] = df_tmp["floor"] + 3
        
        df_tmp = get_pop_rate(df_tmp)
        df_tmp = categorize(df_tmp)

    test = df_tmp[df_tmp["date"].str.contains("2021")].copy()
    data = df_tmp.drop(test.index, axis = 0)
    data = data.reset_index(drop = True)
    test = test.reset_index(drop = True)
    
    target = data["perprice"]
    y_test_true = test["perprice"]
    
    data = data.drop("perprice", axis = 1)
    test = test.drop("perprice", axis = 1)
    
    num_col = data.select_dtypes(exclude = ["object","category"]).columns.to_list()
    
    log_col = ["area", "perhold", "male_for", "female_for", "floor"] 
    cat_col = data.select_dtypes(include = ["object", "category"]).columns.to_list()

    return data, target, test, y_test_true    


# ---------------------------------------------------------------------------------------------
    
def scale_data(X_train, X_valid, X_test):
    tmp_X_train = X_train.copy()
    tmp_X_valid = X_valid.copy()
    tmp_X_test = X_test.copy()
    
    tmp_X_train = tmp_X_train.reset_index(drop = True)
    tmp_X_valid = tmp_X_valid.reset_index(drop = True)
    tmp_X_test = tmp_X_test.reset_index(drop = True)
    
    
    # scale

    
    sd_scaler = StandardScaler()
    rb_scaler = RobustScaler()
    
    tmp_X_train[num_col] = sd_scaler.fit_transform(tmp_X_train[num_col])
    tmp_X_valid[num_col] = sd_scaler.transform(tmp_X_valid[num_col])
    tmp_X_test[num_col] = sd_scaler.transform(tmp_X_test[num_col])
    
    
    final_col = [
#                  'date', 
                 'gu', 
                 'dong', 
                 'area',
                 'floor',
#                  'built',
#                  'doro',
                 'tradetype', 
                 'interest',
#                  'growth',
                 'unemployment',
                 'inflation',
                 'stock',
                 'house_debit',
                 'ex_dollar',
#                  'ex_yen',
                 'household',
                 'pop',
#                  'male_kor',
#                  'female_kor',
#                  'male_for',
#                  'female_for',
                 'rate_male',
                 'rate_female',
                 'rate_male_f', 
                 'rate_female_f',
                 'rate_senior',
                 'perhold',
                 'senior',
#                  'dong_label',
                 'doro_trans',
#                  'year_trans',
                 'year_gap',
                 'floor_level',
                 'oil_price'
                ]
    return tmp_X_train[final_col], tmp_X_valid[final_col], tmp_X_test[final_col]

    
if __name__ == '__main__':
    split_test_train(df, preprocess = bool)