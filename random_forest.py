import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan

import requests
import json


df = pd.read_csv('27_11_2021.csv')

df = df[~(df == 0).any(axis=1)]


LABEL_VAL_THRESH = 50

filtered_indices = []
for k, d in df.groupby(["EPPOCode"]):  # k is uinique label name, d corresponds subset for each unique label
    d1 = d.EPPOCode.value_counts() > LABEL_VAL_THRESH  # d1 is a tuple where 1st key, boolean flag
#     print(d1)
    if d1.values[0]:
        #print("key - ", d.label.value_counts().keys()[0], "value -", d.label.value_counts().values[0])
        filtered_indices.extend(d.index)

# filtered_df = df.loc[df.index[filtered_indices]]
filtered_df = df.loc[filtered_indices]
filtered_df = filtered_df.reset_index(drop=True)

filtered_df.reset_index(drop=True, inplace=True)

#p = filtered_df[filtered_df.EPPOCode != 'UNKDICO'  ] to drop rows across single string
#filtered_df = filtered_df[~filtered_df['EPPOCode'].isin(['UNKDICO','BRSNN', '1GRAF', 'HORVX','VICFX','1PLAK','UNKMONO','SINAL','BEAVX','GLXMA','BEAVA','1IPOG','1CRUF','1AMAF','PIBSX','3CERC','1CUCF','ZEAMX','GOSHI','1ERIG','1LEGF','1FOPG','ERICA','ORYSA','1AMAF','LYPES','URTUR','CPSFR','PTNHY','LACSE','TAROF','1BRSG','GOSHI','LAMPU','1GASG','1CPSG','1CYPF','CASTO','SENVU','1PANG','SINAR','CIEAR','1CHEG','VERAR','1URTG','SOLTU','BIDPI','COMBE','CENCY','FUMOF','DESSO','ABUTH','VERPE','POLLA','LAMAM','SOLNI','ALLCE'])]
filtered_df = filtered_df[~filtered_df['EPPOCode'].isin(['UNKDICO','BRSNN', '1GRAF', 'HORVX','VICFX','1PLAK',
                                      'UNKMONO','SINAL','BEAVX','GLXMA','BEAVA','1IPOG','1CRUF','1AMAF','PIBSX',
                                      '3CERC','1CUCF','ZEAMX','GOSHI','1ERIG','1LEGF','1FOPG','ERICA'])]


#dropping one of the stage early to get proper balance dataset
filtered_df = filtered_df[filtered_df.growth_state != 'EARLY']

filtered_df.reset_index(drop=True, inplace=True)

#label encoder is mainly made for target column for classification problem but it can be used to encode 
# feature column too, if u r only using Tree type classification
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
filtered_df['EPPOCode'] = le.fit_transform(filtered_df['EPPOCode'])
my_encodings  = dict(zip(le.classes_, range(len(le.classes_))))


from category_encoders import OrdinalEncoder

mapping = [{'col': 'growth_state',
            'mapping': pd.Series( data = {'SEEDLING/EMERGED':1, 'MID':0,
                          'ADVANCED/TILLERING':0, 'BRANCHING/ELONGATION':0, 'FLOWERING':0 }),
           'data_type': object }]

enc = OrdinalEncoder(cols=['growth_state'], mapping=mapping)

filtered_df['growth_state'] = enc.fit_transform(filtered_df['growth_state'])

la = []
lo = []
for index, row in filtered_df.iterrows():
    temp = row['geoCord'].split(",")
    lat,lon = temp[0],temp[1]
    la.append(lat)
    lo.append(lon)
    
filtered_df['latitude'] = la
filtered_df['longitude'] = lo
    

filtered_df  = filtered_df[['imageDate', 'latitude', 'longitude', 'soilT', 'soilM',
                            'EPPOCode', 'growth_state']]

filtered_df['latitude'] = filtered_df['latitude'].astype(float)
filtered_df['longitude'] = filtered_df['longitude'].astype(float)

# Converting to date object (str) to datetime stamp
filtered_df['imageDate']  = pd.to_datetime(filtered_df['imageDate'])

#converting datetime stamp to integer values (eg: 21-04-2021 to day of a year )
# filtered_df['imageDate'] = filtered_df['imageDate'].dt.day  #will only give day of month
filtered_df['imageDate'] = filtered_df['imageDate'].dt.dayofyear

filtered_df['soilT'] = np.around(filtered_df['soilT'], 4)
filtered_df['soilM'] = np.around(filtered_df['soilM'], 4)

#Model Building Phase
#importing all necessary common libraries 
#importing and using of confusion_matrix methods
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


#splitting the dataset into training set and test set
X = filtered_df.iloc[:, :-1].values
y = filtered_df.iloc[:, -1].values

#method 1 for imbalance dataset using train_test_split(stratify = target)
#We should try to make sure our splits accurately represent the distribution of our target variable. A very 
#simple way to do this is to use the stratify parameter when calling the train_test_split function.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42, stratify = y )


# Model 2
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 150,criterion = 'entropy',
                                  max_features = 'auto',random_state = np.random.seed(19),
                                  max_depth = 10)
rf_model.fit(X_train,y_train)

y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)
accuracy_score(y_test, y_pred)


print(f1_score(y_test, y_pred)) 
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred)) 

fig, ax = plt.subplots(figsize=(8, 8))
plot_confusion_matrix(rf_model, X_test, y_test, ax=ax)  
plt.show()


#saving the model
import pickle

output = open('rf_class_model2.pkl', 'wb', errors='ignore')
pickle.dump(rf_model, output)
output.close()



