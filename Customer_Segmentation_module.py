# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:21:43 2022

@author: HP
"""

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization


class EDA:
    def displot_graph(self,con_col,df):
        # continuous
        for i in con_col: 
            plt.figure()
            sns.distplot(df[i])
            plt.show()  
            
    
    def countplot_graph(self,cat_col,df):
        for i in cat_col: 
            plt.figure()
            sns.countplot(df[i])
            plt.show()  
            
            
class ModelDevelopment:
    def simple_dl_model(self,X_shape,nb_class,nb_node=128,dropout_rate= 0.3):
        
        model = Sequential() # to creat a container 
        model.add(Input(shape = X_shape))
        model.add(Dense(nb_node, activation ='relu', name = '1st_hidden_layer')) #sigmoid #relu
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node, activation ='relu', name = '2nd_hidden_layer'))#sigmoid #relu
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class, activation= 'softmax')) # output layer
        model.summary()
        
        return model
    
    
