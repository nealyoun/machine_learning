import pymysql
import pandas as pd




host = "mudatabase.cr4neufosdwe.ap-northeast-2.rds.amazonaws.com"
port = 3306
username = "mu"
database = "mudatabase"
password = "munanodb"

query = """

    select
        aa.date,
        aa.gu,
        aa.dong,
        aa.area,
        aa.floor,
        aa.built,
        aa.doro,
        aa.tradetype,
        aa.price,
        aa.interest,
        aa.growth,
        aa.unemployment,
        aa.inflation,
        aa.stock,
        aa.house_debit,
        aa.ex_dallar as ex_dollar,
        aa.ex_yen,
        aa.household,
        aa.pop,
        aa.male_kor,
        aa.female_kor,
        aa.male_for,
        aa.female_for,
        aa.perhold,
        aa.senior,
        bb.oil as oil_price
        from
            (select
                date_format(a.date, "%Y-%m") as date,
                a.gu,
                a.dong,
                a.area,
                a.floor,
                a.built,
                a.doro,
                a.tradetype,
                a.price,
                b.interest,
                b.growth,
                b.unemployment,
                b.inflation,
                b.stock,
                b.house_debit,
                b.ex_dallar,
                b.ex_yen,
                c.household,
                c.pop,
                c.male_kor,
                c.female_kor,
                c.male_for,
                c.female_for,
                c.perhold,
                c.senior
            from
                sell_re as a
            left join
                eco_macro as b
            on
                date_format(a.date, "%Y-%m") = date_format(b.date, "%Y-%m")
            left join
                population as c
            on
                date_format(a.date, "%Y-%m") = date_format(c.date, "%Y-%m")
                and a.dong = c.dong) as aa
        left join mudatabase.oil as bb
        on
            aa.date = date_format(bb.date, "%Y-%m")
            and aa.dong = bb.dong
    ;

    """






def connect_rds(host, port, username, database, password):
    conn = pymysql.connect(host = host , user = username, passwd = password, db = database, port = port,
                           use_unicode= True, charset= "utf8")

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    return conn, cursor


def load_data(query):
    conn, cursor = connect_rds(host, port, username, database, password)
    cursor.execute(query)
    result = cursor.fetchall()
    df = pd.DataFrame(result)
    
    return df



if __name__ == '__main__':
    load_data()