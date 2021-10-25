from flask import render_template, request
from flask import Flask, abort, jsonify
import requests
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/kakaoadd", methods = ['POST'])
def searchadd() :
    
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    header = {'Authorization': 'KakaoAK d60530ee8072efab8c5dbcd42e7f95cd'}
    query = {'query' : request.form['search']}
    response = requests.get(url, headers=header, params=query)
    
    try :
        if response.ok :
            result = response.json()
            if result['meta']['total_count'] > 0:
                adds = {'result' : True,
                        'gu' : result['documents'][0]['address']['region_2depth_name'],
                        'dong' : result['documents'][0]['address']['region_3depth_h_name'],
                        'doro' : result['documents'][0]['road_address']['road_name']
                       }
                return adds
            else :
                return {'result' : False}
        else :
            return {'result' : False}
    except :
        print("에러")
        
@app.route('/predictValue', methods=['POST'])
def predict_value() :
    import predict_output as po
    
    input_data = {}
    input_data['gu'] = request.form['predgu']
    input_data['dong'] = request.form['preddong']
    input_data['doro'] = request.form['preddoro']
    input_data['area'] = float(request.form['predarea'])
    input_data['floor'] = int(request.form['predfloor'])
    input_data['tradetype'] = int(request.form['predtype'])
    input_data['year'] = int(request.form['predyear'])
    input_data['credit'] = int(request.form['predcredit'])
    input_data['pay'] = int(request.form['predpay'])
    
    result = po.plot_output(input_data)
    
    return jsonify(result)
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)