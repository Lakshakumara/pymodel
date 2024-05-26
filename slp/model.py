from __future__ import print_function
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#tensor

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# clustering
import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
   
from slp.oulad import OULAD_data
import pickle
import numpy as np
import pandas as pd
import logging
import sys
import os
import json


import matplotlib.pyplot as plt
import seaborn as sns

class Model:
    def __init__(self):
        self.showData = False
        self.model=None
        self.ou = OULAD_data()
    
    def get_classifier(self, X, y):
        """Returns a logistic regression classifier after cross-validation"""

        solver = 'liblinear'
        multi_class = 'ovr'

        if hasattr(self, 'C') is False:

            # Cross validation - to select the best constants.
            lgcv = LogisticRegressionCV(solver=solver, multi_class=multi_class)
            lgcv.fit(X, y[:, 0])

            if len(lgcv.C_) == 1:
                C = lgcv.C_[0]
            else:
                # Chose the best C = the class with more samples.
                # Ideally multiclass problems will be multinomial.
                [_, counts] = np.unique(y[:, 0], return_counts=True)
                C = lgcv.C_[np.argmax(counts)]
                logging.info('From all classes best C values (%s), %f has been selected',
                             str(lgcv.C_), C)
            logging.info("Best C: %f", C)

        return LogisticRegression(solver=solver, tol=1e-1, C=C)



    def dump(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.vocab_size, f)
            pickle.dump(self.vectorizer, f)
            pickle.dump(self.clf, f)
    
    def getWithdraw(self):
        df=self.ou._successORdrop(True)
        df = df.drop(df[df['final_result'] == 'Fail'].index)
        df = df.drop(df[df['final_result'] == 'Pass'].index)
        df = df.drop(df[df['final_result'] == 'Distinction'].index)
        if self.showData:
            print(f"Withdraw\n " ,df)
        return df
    
    def getFail(self):
        df=self.notFails()
        df = df.drop(df[df['final_result'] == 'Pass'].index)
        df = df.drop(df[df['final_result'] == 'Withdrawn'].index)
        df = df.drop(df[df['final_result'] == 'Distinction'].index)
        if self.showData:
            print(f"fails\n " ,df)
        return df
    
    def getPass(self):
        df=self.notFails()
        df = df.drop(df[df['final_result'] == 'Fail'].index)
        df = df.drop(df[df['final_result'] == 'Withdrawn'].index)
        df = df.drop(df[df['final_result'] == 'Distinction'].index)
        if self.showData:
            print(f"Passed\n " ,df)
        return df
    
    def getDistinction(self):
        df=self.notFails()
        df = df.drop(df[df['final_result'] == 'Withdrawn'].index)
        df = df.drop(df[df['final_result'] == 'Fail'].index)
        df = df.drop(df[df['final_result'] == 'Pass'].index)
        if self.showData:
            print(f"Dictinction\n " ,df)
        return df
        
    def notFails(self):
        df=self.ou._successORdrop()
        df=df[df["sum_click"]<=10]
        df=df[df["num_of_prev_attempts"]<=4]
        if self.showData:
            print(f"notFails\n " ,df)
        return df
    
    
    def train(self, args0, args1):
        #withdeaw+fails 65%
        #dff=self.getWithdraw()
        #dfp=self.getFail()
        
        #withdeaw+pass 98%
        #dff=self.getWithdraw()
        #dfp=self.getPass()
        
        #Fail+pass 89%
        #dff=self.getFail()
        #dfp=self.getPass()
        
        #Pass+Distinction 84%
        #dff=self.getPass()
        #dfp=self.getDistinction()
        
        #Fail+Distinction 92%
        dff=self.getFail()
        dfp=self.getDistinction()
        
        #Fail+Pass+Distinction 79%
        #dff=pd.concat([self.getFail(), self.getPass()])
        #dfp=self.getDistinction()
        
        #Withdrawn+Distinction 99%
        #dff=self.getWithdraw()
        #dfp=self.getDistinction()
        
        
        df = pd.concat([dff, dfp])
        df = df.drop(df[df['weighted_grade'] == 0].index)
        df.drop([ "id_assessment","code_module","code_presentation","id_student","pass","num_of_prev_attempts"], axis=1, inplace=True)
        
        
        print("\nFinal Model\n\n", df)

        #df = df.drop(df[df['score_rate'] == 0].index)
 
        #df.to_csv('./data/modelx.csv', sep=',', index=True, encoding='utf-8')
        X=df.drop("final_result", axis=1)
        y=df["final_result"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        scaler1=MinMaxScaler()
        X1_test=X_test
        X1_train=X_train
        
        X1_train=scaler1.fit_transform(X1_train)
        X1_test=scaler1.transform(X1_test)
        
        self.model = LogisticRegression(max_iter=10000)
        self.model.fit(X1_train, y_train)
        result_lr1=self.model.predict(X1_test)
        #print(confusion_matrix(y_test,result_lr1))
        #print("\n")
        r=classification_report(y_test,result_lr1)
        print(r)
        #p=np.array([[82.4,82]])
        #y_pred = self.model.predict(p)
        #print(y_pred)
        return r

    def trainRF(self):
        #withdeaw+fails 64%
        #dff=self.getWithdraw()
        #dfp=self.getFail()
        
        #withdeaw+pass 98%
        #dff=self.getWithdraw()
        #dfp=self.getPass()
        
        #Fail+pass 90%
        #dff=self.getFail()
        #dfp=self.getPass()
        
        #Pass+Distinction 86%
        #dff=self.getPass()
        #dfp=self.getDistinction()
        
        #Fail+Distinction 97%
        #dff=self.getFail()
        #dfp=self.getDistinction()
        
        #Fail+Pass+Distinction 80%
        #dff=pd.concat([self.getFail(), self.getPass()])
        #dfp=self.getDistinction()
        
        #Withdrawn+Distinction 99%
        #dfw=self.getWithdraw()
        dff=self.getFail()
        dfp=self.getPass()
        dfd=self.getDistinction()
        
        
        df = pd.concat([dff,dfp, dfd])
        #dfwg0=df[df["weighted_grade"] == 0]
        #dfwg0.drop([ "id_assessment","code_module","code_presentation","id_student","pass","num_of_prev_attempts"], axis=1, inplace=True)
        
        df = df.drop(df[df['weighted_grade'] == 0].index)
        df.drop([ "id_assessment","code_module","code_presentation","id_student","pass","num_of_prev_attempts","sum_click"], axis=1, inplace=True)
        
        df = df.loc[~((df['final_result'] == 'Pass') & (df['weighted_grade'] < 25))]
        
        """ _summary_plt.figure(figsize=(8,6))
        sns.countplot(data=df, x="final_result")
        sns.heatmap(df.corr(), annot=True, linewidth=.5)
        sns.pairplot(df)
        plt.show()
        """
       
        
        
        print("\nFinal Model\n\n", df)

        #df = df.drop(df[df['score_rate'] == 0].index)
        
        #df.to_csv('./data/final_model.csv', sep=',', index=False, encoding='utf-8')
        X=df.drop("final_result", axis=1)
        y=df["final_result"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler1=MinMaxScaler()
        X1_test=X_test
        X1_train=X_train
        
        X1_train=scaler1.fit_transform(X1_train)
        X1_test=scaler1.transform(X1_test)
        
        self.model = RandomForestClassifier(n_estimators=300)
        self.model.fit(X1_train,y_train)
        # save the Model
        
        #self.serialize("rf_fpd")
        
        # end of saving
        result_rf1=self.model.predict(X1_test)
        print(confusion_matrix(y_test,result_rf1))
        print("\n")
        print(classification_report(y_test,result_rf1))
    
    def kmean(self):
        pass
                   
    def train1(self):
        df=self.ou._successORdrop()
        
        df=df[df["sum_click"]<=10]
        df=df[df["num_of_prev_attempts"]<=4]
        
        df.drop([ "date"], axis=1, inplace=True)
        df.drop("id_assessment", axis=1, inplace=True)
        df.drop("pass", axis=1, inplace=True)
        df.drop("num_of_prev_attempts", axis=1, inplace=True)
        
        df = df.drop(df[df['weighted_grade'] == 0].index)
        
        #df=df.drop("final_result", axis=1)
        df = df.drop(df[df['final_result'] == 'Withdrawn'].index)
        if self.showData:
            print("Final Model\n", df)
        X=df.drop("final_result", axis=1)
        y=df["final_result"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        scaler1=MinMaxScaler()
        #scaler2=MinMaxScaler()
        #scaler3=MinMaxScaler()
        #1 contains both, 2 just pass_rate e 3 just weighted_grade
        X1_test=X_test
        X1_train=X_train
        
        X1_train=scaler1.fit_transform(X1_train)
        X1_test=scaler1.transform(X1_test)
        
        self.model = LogisticRegression(max_iter=10000)
        self.model.fit(X1_train, y_train)
        result_lr1=self.model.predict(X1_test)
        print(confusion_matrix(y_test,result_lr1))
        print("\n")
        print(classification_report(y_test,result_lr1))
        
        
        #p=np.array([[82.4,82]])
        #y_pred = self.model.predict(p)
        #print(y_pred)
    
    def serialize(self, fname):
        path=sys.path.append('/dump/model/rf/'+fname+".pickle")
        #ff = path+fname+".pickle"
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def predictionRF(self, fname, predictor):
        #if os.path.isdir(persistencedir) is False:
        #    if os.makedirs(persistencedir) is False:
        #        raise OSError('Directory ' + persistencedir + ' can not be created.')
            
        #print(persistencedir)
        
        absolute_path = os.path.dirname(__file__)
        relative_path = "dump\\model\\rf\\"+fname+".pickle"
        full_path = os.path.join(absolute_path, relative_path)
        
        with open(full_path, 'rb') as f:
            self.model = pickle.load(f)
            
        scaler1=MinMaxScaler()
        p=np.array([predictor])
        #p=scaler1.transform(p)
        y_pred = self.model.predict(p)
        y_proba = self.model.predict_proba(p)
        
        # Probabilities of the predicted response being correct.
        probabilities = y_proba[range(len(y_proba)), 0]

        result = dict()
        result['inputs']=predictor
        result['status'] = 0
        result['info'] = []
        # First column sampleids, second the prediction and third how
        # reliable is the prediction (from 0 to 1).
        result['predictions'] = np.vstack((0, y_pred, probabilities)).T.tolist()
        
        print(json.dumps(result))
        sys.exit(result['status'])
    
    def x(self, fname, predictor):
        absolute_path = os.path.dirname(__file__)
        relative_path = "dump\\model\\rf\\"+fname+".pickle"
        full_path = os.path.join(absolute_path, relative_path)

        persistencedir = os.path.join('./dump','model','rf')
        print(full_path)
        if os.path.exists(full_path):
            x="path exists"
        else :
            x="not path"
        return x, full_path
           
    def predict(self, X):
        #X = self.vectorizer.transform(X)
        y_pred = self.model.predict(X)
        return 
    
    def categories(self, cat):
        if cat=="Fail":
            return 0
        if cat=="Pass":
            return 1
        if cat=="Distinction":
            return 1
    
    
    def tf(self):
        #dfw=self.getWithdraw()
        dff=self.getFail()
        dfp=self.getPass()
        dfd=self.getDistinction()
        
        df = pd.concat([dff,dfp, dfd])
        df = df.drop(df[df['weighted_grade'] == 0].index)
        df.drop([ "id_assessment","code_module","code_presentation","id_student","pass","num_of_prev_attempts"], axis=1, inplace=True)
        
        if self.showData:
            print("\nFinal Model\n\n", df)

        #df = df.drop(df[df['score_rate'] == 0].index)
 
        #df.to_csv('./data/modelx.csv', sep=',', index=True, encoding='utf-8')
        X=df.drop("final_result", axis=1)
        y=df["final_result"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        scaler1=MinMaxScaler()
        
        X1_test=X_test
        X1_train=X_train
        
        X1_train=scaler1.fit_transform(X1_train)
        X1_test=scaler1.transform(X1_test)
        
        model1=Sequential()
        model1.add(Dense(6, activation="relu"))
        model1.add(Dropout(0.5))
        model1.add(Dense(3, activation="relu"))
        model1.add(Dense(1, activation="sigmoid"))

        model1.compile(loss="binary_crossentropy", optimizer="adam")
        y_test=list(map(self.categories,y_test))
        y_train=list(map(self.categories,y_train))
        
        y_train=np.asarray(y_train)
        y_test=np.asarray(y_test)
        early_stop=EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
        
        model1.fit(x=X1_train, y=y_train, epochs=2000, validation_data=(X1_test,y_test),callbacks=[early_stop])
        losses=pd.DataFrame(model1.history.history)
        losses.plot()
        
        predictions=model1.predict(X1_test)
        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test,predictions))
        
        