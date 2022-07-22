# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:26:24 2022

@author: HP
"""
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer

from Customer_Segmentation_module import ModelDevelopment
from Customer_Segmentation_module import EDA

import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import numpy as np
import datetime
import pickle
import os


#%% constant
CSV_PATH = os.path.join(os.getcwd(),'Train.csv')
MMS_PATH = os.path.join(os.getcwd(),'Models','mms.pkl')
LOGS_PATH = os.path.join(os.getcwd(),'logs', datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
OHE_SAVE_PATH = os.path.join(os.getcwd(),'Models','ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Models','model.h5')

#%%

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% STEP 1  DATA LOADING
os.getcwd()
df = pd.read_csv(CSV_PATH)

#%% STEP 2 DATA VIZUALIZATION

df.describe()# to obtain basic statistics
df.boxplot()
df.info()
df.isna().sum()

df = df.drop(labels= ['id'], axis= 1)

cat_col = df.columns[df.dtypes == 'object']
con_col = df.columns[(df.dtypes=='int64') | (df.dtypes == 'float64')]

eda = EDA()
eda.displot_graph(con_col, df)
eda.countplot_graph(cat_col, df)

     
df = df.drop(labels= ['term_deposit_subscribed','days_since_prev_campaign_contact'], axis= 1) 

cat_col = df.columns[df.dtypes == 'object']
con_col = df.columns[(df.dtypes=='int64') | (df.dtypes == 'float64')]

df.describe().T
df.duplicated().sum()

#%% Step 3) Data Cleaning

df.isna().sum()

df.describe().T

#%% KNNImputer()
# conversion into number is required

le = LabelEncoder()

for i in cat_col:
    temp = df[i]
    temp[temp.notnull()]= le.fit_transform(temp[temp.notnull()])
    df[i] = pd.to_numeric(temp,errors = 'coerce')
    ENCODER_SAVE_PATH = os.path.join(os.getcwd(),'Models', i+'_encoder.pkl')
    pickle.dump(le,open(ENCODER_SAVE_PATH,'wb'))

imputer = KNNImputer()
imputed_df = imputer.fit_transform(df) # returns array
imputed_df = pd.DataFrame(imputed_df) # To convert array into DataFrame format
imputed_df.columns = df.columns # To add column names
imputed_df.isna().sum() 

for i in cat_col:
    imputed_df[i] = np.floor(imputed_df[i]).astype(int)
    

imputed_df.duplicated().sum()

#%% step 4) feature selection

# cont vs cat
# logistic regression

X = imputed_df.drop(labels= 'prev_campaign_outcome', axis = 1)
y = imputed_df['prev_campaign_outcome']

selected_features = []

for i in con_col:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(X[i], axis = -1),y) # X: continuous  Y: catagorical
    print(i)
    print(lr.score(np.expand_dims(X[i], axis = -1),y))
    if lr.score(np.expand_dims(X[i], axis = -1),y) > 0.8:
        selected_features.append(i)
        
print(selected_features)
    
#%%   
# cat vs cat

for i in cat_col:
    print(i)
    matrix = pd.crosstab(imputed_df[i],y).to_numpy() 
    print(cramers_corrected_stat(matrix))
    if cramers_corrected_stat(matrix) > 0.1:
        selected_features.append(i)
 
print(selected_features)

imputed_df = imputed_df.loc[:,selected_features]
X = imputed_df.drop(labels= 'prev_campaign_outcome', axis = 1)
y = imputed_df['prev_campaign_outcome']

#%% Step 5)-Pre-processing

#MMS

mms = MinMaxScaler()
X = mms.fit_transform(X)

with open(MMS_PATH, 'wb') as file:
    pickle.dump(mms,file)

#OHE

ohe = OneHotEncoder(sparse = False)
y = ohe.fit_transform(np.expand_dims(y,axis = 1))

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, 
                                                  random_state=123)

#%% Model devalopment

X_shape=np.shape(X_train)[1:]
nb_class = len(np.unique(y,axis=0))

md = ModelDevelopment()
model = md.simple_dl_model(X_shape,nb_class,nb_node = 40 ,dropout_rate = 0.3)

plot_model(model,show_shapes=True,show_layer_names= True)

model.compile(optimizer= 'adam', 
              loss = 'categorical_crossentropy',
              metrics='acc')

#callback

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

early_callback = EarlyStopping(monitor = 'val_loss', patience = 3)

hist= model.fit(X_train,y_train,
                epochs= 50, 
                validation_data=(X_test,y_test),
                callbacks = [early_callback,tensorboard_callback])


#%% MODEL EVALUATION

print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Training loss','Validation Loss'])
plt.show()


plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['Training acc','Validation Acc'])
plt.show()

#%% 
y_pred =np.argmax(model.predict(X_test),axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test,y_pred))


#%% model saving

#OHE
with open(OHE_SAVE_PATH, 'wb') as file:
    pickle.dump(ohe,file)
        
# Model
model.save(MODEL_SAVE_PATH)

