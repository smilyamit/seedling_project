from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('rf_clf_model.pkl', 'rb', errors='ignore'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emerg_pred_api',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # float_features = [float(x) for x in request.form.values()]
    features = [x for x in request.form.values()]
    if isinstance(features[5], str):
        weed_enc = {'1AMAG': 0, '1ATXG': 1, '1CARG': 2, '1COMF': 3, 
               '1CYPG': 4, '1DIGG': 5, '1MYOG': 6, '1TRFG': 7, '1VERG': 8, 
            '1VIOG': 9, 'AMARE': 10, 'AMBEL': 11, 'AVESA': 12, 'BRSNW': 13,
            'CAPBP': 14, 'CARHI': 15, 'CHEAL': 16, 'CHEPO': 17, 'CIRAR': 18,
            'CPSAN': 19, 'DIGSA': 20, 'EQUAR': 21, 'GALAP': 22, 'GASPA': 23,
            'GERDI': 24, 'IPOCC': 25, 'IPOHE': 26, 'IPOHG': 27, 'IPOLA': 28,
            'IPOTR': 29, 'KCHSC': 30, 'MATCH': 31, 'MYOAR': 32, 'PLAMA': 33,
            'POLAV': 34, 'POLCO': 35, 'POLPE': 36, 'POROL': 37, 'RANRE': 38,
            'STEME': 39, 'THLAR': 40, 'VIOAR': 41}
        for k, v in weed_enc.items():
            if k == features[5]:
                features[5] = weed_enc[k]

        final_features = [np.array(features)]
        prediction = model.predict(final_features)

    else:
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
    
    if prediction == [1]:
        output = 'Seedling'
    elif prediction == [0]:
        output = 'Other'

    return render_template('index.html', prediction_text='Weed emergence stage is => {}'.format(output))

@app.route('/data_api')
def predict_api():
    # my_data = {"daysInYear":291,
    #          "lat":49.411743,
    #          "long": 8.399428,
    #          "soilTemp":14.271,
    #          "soilMoist":0.420,
    #          "eppo_code":14}
    # prediction = model.predict([np.array(list(my_data.values()))])

    # if prediction == [1]:
    #     output = 'Seedling stage'
    # elif prediction == [0]:
    #     output = 'Other Stage'
    df = pd.read_csv('dataForClfr2.csv')
    my_data = []
    dic_d = {}
    for i in range(len(df)):
        my_data.append(df.iloc[i].to_dict())
        dic_d['data'] = my_data
        # break

    
    return dic_d
    # return  '{} {}'.format(my_data, output)

if __name__ == "__main__":
    app.run(debug=True)

'''
just for test
y_pred2 = xg_model.predict([[ 239, 47.763432, 19.105979, 22.176, 0.389, 14]])
#1008 	239 	47.763432 	19.105979 	22.176 	0.389 	14 	0

y_pred4 = xg_model.predict([[ 256, 51.053406, 6.911788, 17.791, 0.386, 14]])
#256 	51.053406 	6.911788 	17.791 	0.386 	14 	1
'''