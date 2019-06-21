import pandas as pd 
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import math

class classificationHandler(object):
    #load the data into a dataframe with no preparation
    def __loadDataIntoDataFrame(path):
        dataSet = pd.read_csv(path, sep=';',header=0, encoding='ascii', engine='python')
        #print(transactions.columns.tolist())
        return dataSet
    #remove the nulls 
    def __cleanTheData(path):
        dataSet=classificationHandler.__loadDataIntoDataFrame(path)
        numberofnullrows=len(dataSet[dataSet.isnull()].index)
        #too much cant delete all the null rows rows
        
        #get all information about null values in each column
        null_columns=dataSet.columns[dataSet.isnull().any()]
        
        

        #variable18 has 2145 null values meaning it doesnt has an effect on the classifier
        #remove variable 18 from the dataset
        dataSet=dataSet.drop("variable18",axis=1)
        #print(trainingdata.head()) 
        
        
        #get summary after performing deletetions
        dataSet=dataSet.dropna()
        #get info of the training set

       # trainingdata.info()
        #return the training data shich han no null values
        return dataSet
    
    def __dataToNumeric(path):
        
        dataSet=classificationHandler.__cleanTheData(path)

        #to convert values to numeric values to be used in further processing
        le = preprocessing.LabelEncoder()
        
        #for v1
        le.fit(dataSet['variable1'])
        y=le.transform(dataSet['variable1'])
        dataSet['variable1']=y
        
        #for v2
        le.fit(dataSet['variable2'])
        y=le.transform(dataSet['variable2'])
        dataSet['variable2']=y

        #for v3
        le.fit(dataSet['variable3'])
        y=le.transform(dataSet['variable3'])
        dataSet['variable3']=y

        #for v4
        le.fit(dataSet['variable4'])
        y=le.transform(dataSet['variable4'])
        dataSet['variable4']=y

        #for v5
        le.fit(dataSet['variable5'])
        y=le.transform(dataSet['variable5'])
        dataSet['variable5']=y

        #for v6
        le.fit(dataSet['variable6'])
        y=le.transform(dataSet['variable6'])
        dataSet['variable6']=y

        #for v7
        le.fit(dataSet['variable7'])
        y=le.transform(dataSet['variable7'])
        dataSet['variable7']=y

        #for v8
        le.fit(dataSet['variable8'])
        y=le.transform(dataSet['variable8'])
        dataSet['variable8']=y

        #for v9
        le.fit(dataSet['variable9'])
        y=le.transform(dataSet['variable9'])
        dataSet['variable9']=y

        #for v10
        le.fit(dataSet['variable10'])
        y=le.transform(dataSet['variable10'])
        dataSet['variable10']=y
        #for v12
        le.fit(dataSet['variable12'])
        y=le.transform(dataSet['variable12'])
        dataSet['variable12']=y

        #for v13
        le.fit(dataSet['variable13'])
        y=le.transform(dataSet['variable13'])
        dataSet['variable13']=y

        #for classlabel
        le.fit(dataSet['classLabel'])
        y=le.transform(dataSet['classLabel'])
        dataSet['classLabel']=y

        #datatable is then returned for normalization 
        return dataSet

    
    
    #define some columns
    def __normalizeColumns(path):
        dataSet=classificationHandler.__dataToNumeric(path)
    
        dataSet['variable2']=(dataSet['variable2']-dataSet['variable2'].min())/(dataSet['variable2'].max()-dataSet['variable2'].min())
        dataSet['variable3']=(dataSet['variable3']-dataSet['variable3'].min())/(dataSet['variable3'].max()-dataSet['variable3'].min())
        dataSet['variable8']=(dataSet['variable8']-dataSet['variable8'].min())/(dataSet['variable8'].max()-dataSet['variable8'].min())
        dataSet['variable11']=(dataSet['variable11']-dataSet['variable11'].min())/(dataSet['variable11'].max()-dataSet['variable11'].min())
        dataSet['variable14']=(dataSet['variable14']-dataSet['variable14'].min())/(dataSet['variable14'].max()-dataSet['variable14'].min())
        dataSet['variable15']=(dataSet['variable15']-dataSet['variable15'].min())/(dataSet['variable15'].max()-dataSet['variable15'].min())
        dataSet['variable17']=(dataSet['variable17']-dataSet['variable17'].min())/(dataSet['variable17'].max()-dataSet['variable17'].min())


        return dataSet    









    def dataClassification():

        #prepare the training data
        trainingdata=classificationHandler.__normalizeColumns('training.csv')
        testdata=classificationHandler.__normalizeColumns('validation.csv')
        #get all the variable for training
        variable_train=trainingdata.iloc[:, :-1].values
        #get all the class label for training
        classlabel_train=trainingdata.iloc[:,17].values
        #get all the variables for testing
        variable_test=testdata.iloc[:, :-1].values
        #get the class label for testing
        classlabel_test=testdata.iloc[:,17].values


        #perform the classification
        classifier= KNeighborsClassifier()
        #the five value should be checked again
        classifier = classifier.fit(variable_train, classlabel_train) 

        #start prediction
        variable_pred = classifier.predict(variable_test)
        
    
        print(variable_pred)
        #evaluate the algorithm
        acc = accuracy_score(classlabel_test, variable_pred)
        print(acc)

    


       