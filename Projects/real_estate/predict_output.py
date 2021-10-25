import preprocess
import numpy as np
import pandas as pd
import datetime as dt
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
import model

from time import gmtime, strftime
from dateutil.relativedelta import relativedelta

import os
from os.path import join
ROOT_PATH = os.getcwd()

from matplotlib import font_manager, rc
plt.style.use('seaborn')
font_manager.get_fontconfig_fonts()
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_prop = font_manager.FontProperties(fname=font_path, size = 16)
plt.rc('axes', unicode_minus = False)

# import model

def get_input_data(user) : 
    ts = pd.date_range(start = dt.datetime.today().strftime('%Y-%m-%d'), end = None, periods = 24, freq = 'MS')
    
    # 사용자 입력 정보 전달
    df = pd.DataFrame()
    df['date'] = ts
    df['gu'] = user["gu"]
    df['dong'] = user['dong']
    df['area'] = user['area']
    df['floor'] = user['floor']
    df['tradetype'] = user['tradetype']
    df['doro_trans'] = preprocess.get_road(user['doro'])
    df['year_gap'] = df['date'].apply(lambda x : x.year - user['year'] +1)
    df['floor_level'] = df['floor'].map(lambda x : preprocess.get_floor_level(x))
    df['date'] = df['date'].apply(lambda x : dt.datetime.strftime(x, '%Y-%m-%d'))
    df['credit'] = user["credit"]
    df['pay'] = user["pay"]
    
    # db 데이터 호출
    conn = pymysql.connect(host='mudatabase.cr4neufosdwe.ap-northeast-2.rds.amazonaws.com', user='mu', password="munanodb",
                           charset='utf8', db='mudatabase', port = 3306 )
    cursor = conn.cursor()
    
    sql = f"""
            select date_format(a.date, '%Y-%m-%d') as date, interest, unemployment, inflation, stock, ex_dallar, 
                    pop, (female_kor / pop) as rate_female, (male_for / pop) as rate_male_f, (female_for / pop) as rate_female_f, (senior / pop) as rate_senior, perhold,
                    oil
            from eco_macro as a
            left join population as b
                on a.date = b.date
            left join oil c
                on a.date = c.date
                    and b.dong = c.dong
                where b.dong = '{user['dong']}'
                    and date_format(a.date, '%Y-%m-%d') between '{df['date'][0]}' and '{df['date'][23]}' ;
            """
    
    cursor.execute(sql)
    db_data = pd.DataFrame(cursor.fetchall())

    cursor.close()
    db_data.columns = ['date', 'interest', 'unemployment', 'inflation', 'stock', 'ex_dollar','pop', 'rate_female', 'rate_male_f', 'rate_female_f', 'rate_senior', 'perhold','oil_price']
    


    df = df.merge(db_data, how='inner', on='date')

    df.loc[:,["area","rate_female","rate_male_f","rate_female_f","rate_senior"]] = df.loc[:,["area","rate_female","rate_male_f","rate_female_f","rate_senior"]].astype("float")
    cat_col = ['gu', 'dong', 'tradetype', 'doro_trans', 'floor_level']
    df[cat_col] = df[cat_col].astype("category")
    
    final_col = ["date","credit","pay",'gu', 'dong', 'area', 'floor', 'tradetype', 'interest', 'unemployment',
            'inflation', 'stock', 'ex_dollar', 'pop', 'rate_female', 'rate_male_f',
            'rate_female_f', 'rate_senior', 'perhold', 'doro_trans', 'year_gap',
            'floor_level', 'oil_price']
    
    df = df[final_col]

    global num_col
    num_col = ['area','floor','interest','unemployment','inflation','stock','ex_dollar','pop','rate_female',
               'rate_male_f','rate_female_f','rate_senior', 'perhold', 'year_gap', 'oil_price']
    
    return df[final_col]





def predict(input_data):
    import model

    df = input_data.copy()
    df = df.drop(["date","pay","credit"], axis = 1)

    model_0, model_1, model_2, model_3, model_4, model_5 = model.load_model()
    scaler_0, scaler_1, scaler_2,scaler_3, scaler_4, scaler_5 = model.load_scaler()

    lst_model = [model_0, model_1, model_2, model_3, model_4, model_5]
    lst_scaler = [scaler_0, scaler_1, scaler_2,scaler_3,scaler_4,scaler_5]

    oof_pred = np.zeros([df.shape[0], ])
    for i in range(len(lst_model)):
        df[num_col] = lst_scaler[i].transform(input_data[num_col])
        oof_pred += lst_model[i].predict(df) 

    output = oof_pred / len(lst_model)
    
    return output, pd.concat([input_data, pd.Series(output, name = "output")], axis =1)

def plot_output(df):
    global output_df

    input_data = get_input_data(df)
    output, output_df = predict(input_data)
    
    # 전세 전환율 5%
    standard_1 = output_df['credit'] + (output_df['pay']*12/0.05)
    # 전세 전환율 2.5% // 법정 전환율
    standard_2 = output_df['credit'] + (output_df['pay']*12/0.025)
    # 집값의 70% 선
    output_df['deadline'] = output_df['area'] * output_df['output'] * 0.7

    # plt.title('')
    plt.figure(figsize = (16,8))
    
    plt.xticks(rotation=45, fontsize = 12)
    plt.xlabel(' ')
    plt.yticks(font_properties = font_prop, fontsize = 12)
    plt.ylabel('가격(만원)', loc='top', font_properties = font_prop, rotation = 0)
    sns.lineplot(data = output_df, x = "date", y = input_data["area"] * output, label ='주택 예측 가격')
    y_axis = (output_df["output"] * output_df["area"]).mean()
    plt.ylim(y_axis * 0.5, y_axis * 1.2 )

    sns.lineplot(data = output_df, x = 'date', y=standard_1, label ='전세가율(월세전환율 기준 5%, 서울 평균)')
#     sns.lineplot(data = output_df, x = 'date', y=standard_2, label='월세전환율 2.5%, 법정 기준')
    sns.lineplot(data = output_df, x = 'date', y="deadline", label='주택 가격 70% 선')
   
    plt.legend(prop = font_prop)
    plt.title('집값 변화 예측 결과', font_properties=font_prop, fontsize = 24)
#     plt.show()
    
    output_plot_id = 'output_plot_{0}.png'.format(strftime("%Y%m%d_%H%M%S", gmtime()))
    file_path = os.path.join(ROOT_PATH, 'static', 'result_img', output_plot_id)
    plt.savefig(file_path, facecolor='#eeeeee', )
    
    dict_pred = { 'file' : output_plot_id,
                  'pred_1' : np.round_(standard_1[0] / (output_df.loc[0,"area"]* output_df.loc[0,"output"]) ,2)*100,
                  'pred_2' : np.round_(standard_1[0] / (output_df.loc[output_df.shape[0]-1,"area"] * output_df.loc[output_df.shape[0]-1,"output"]),2)*100
                }
    
    result_to_db(df, output)
    
    return dict_pred


# 서버 DB 에 검색 결과 누적하기
def result_to_db(df, output) :
    
    # db 연결
    conn = pymysql.connect(host='mudatabase.cr4neufosdwe.ap-northeast-2.rds.amazonaws.com', user='mu', password="munanodb",
                           charset='utf8', db='mudatabase', port = 3306 )
    cursor = conn.cursor()
    
    
    sql = """
            INSERT INTO predResult (gu, dong, doro, area, floor, tradetype, built, credit, pay,
                                    p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          """

    data = []

    s1 = df['gu']
    s2 = df['dong']
    s3 = df['doro']
    s4 = df['area']
    s5 = df['floor']
    s6 = df['tradetype']
    s7 = df['year']
    s8 = df['credit']
    s9 = df['pay']
    
    data = [s1, s2,s3,s4,s5,s6,s7,s8,s9]
    
    for i in range(len(output)) :
        data.append(df['area'] * output[i])
    
    cursor.execute(sql, data)
    conn.commit()

if __name__ == "__main__":
    plot_output()