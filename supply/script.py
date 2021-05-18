
import pandas as pd
import os
from dfply import * # https://github.com/kieferk/dfply#rename
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression



# def set_wd(wd="C:/Users/Eliot Tabet/Desktop/ENSTA/"):
#    os.chdir(wd)


#1) import consumption data from DE.csv into a pandas DataFrame and rename Date (CET) column to Date
#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_xlsx(f_name = "storage_data.xlsx", delimeter = ";"):
    data=pd.ExcelFile(f_name)
    SheetNameList = data.sheet_names
    storage_data = {sheet: data.parse(sheet) for sheet in SheetNameList}
    return storage_data

def importPriceData(fName='price_data.csv'):
	priceData=pd.read_csv(fName, sep=";")
	priceData=priceData.dropna() >> mutate(Date = pd.to_datetime(priceData['Date'], format = "%d/%m/%Y"))
	priceData.rename(columns={'Date' : 'gasDayStartedOn' }, inplace=True)
	return priceData


class fichierExcel:
   def __init__(self, fichier):
      self.allNameSheet=fichier.keys()
      self.d = fichier
      self.dictionaire1=dict()
      self.dictionaire2=dict()
      self.listSheet=[]
      
   def collectAllRegression(self,priceData):
   		self.compteurRandom = 0;
   		self.compteurLogistique = 0;
   		for i in self.allNameSheet:
   			f = self.d[i]
   			st = sheet(i, f)
   			st.createColumn(priceData)
   			st.regressionLogistique()
   			st.regressionRandom()
   			st.regressionLineaire()
   			if (st.bestRegression()=="random"):
   				self.compteurRandom+=1
   			else:
   				self.compteurLogistique+=1
   			self.listSheet.append(st)
   		if (self.compteurLogistique>=self.compteurRandom):
   			print("Best regression is RandomForest")

   	# def bestRegressionOverAll(self):
   	# 	self.compteurRandom = 0;
   	# 	self.compteurLogistique = 0;
   	# 	for i in self.allNameSheet:




   		
   		

	

class sheet:

	def __init__(self, dataFrameName, storage_data):
		self.df = storage_data
		self.dfname=dataFrameName
		self.len=len(self.df.full)

	def createColumn(self,priceData):
		
		self.newdf=self.df >> select(X.gasDayStartedOn,X.full,X.injection,X.withdrawal)
		self.newdf['NW']=self.newdf['withdrawal']-self.newdf['injection']
		self.newdf['lagged_NW']=self.newdf.NW.shift(1)
		self.newdf['Nwithdrawal_binary'] = np.where(self.newdf['NW']>0, 1, 0)
		self.newdf['FSW1']=np.where((self.newdf['full']-45)>0,self.newdf['full']-45,0)
		self.newdf['FSW2']=np.where((45-self.newdf['full'])>0,45-self.newdf['full'],0)
		self.newdf=self.newdf >> select(X.gasDayStartedOn,X.NW,X.lagged_NW,X.Nwithdrawal_binary,X.FSW1,X.FSW2)
		self.newdf=pd.merge(self.newdf,priceData, on='gasDayStartedOn')
		


	def regressionLogistique(self):

		self.Y = self.newdf.Nwithdrawal_binary
		self.X = self.newdf >> select(X.lagged_NW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_NCG,X.SAS_NBP)
		self.x_train_lr, self.x_test_lr,self.y_train_lr,self.y_test_lr=train_test_split(self.X,self.Y)
		self.lr = LogisticRegression()
		self.lr.fit(self.x_train_lr,self.y_train_lr)
		self.y_predLogistique=self.lr.predict(self.x_test_lr)
		# print(self.lr.coef_)
		# print(self.lr.intercept_)
		self.confusion_lr=confusion_matrix(self.y_test_lr,self.y_predLogistique)
		# print(self.confusion)
		self.probs_lr=self.lr.predict_proba(self.x_test_lr)[:,1]
		cm=self.confusion_lr
		self.dictMetricLogistique={'recall': recall_score(self.y_test_lr, self.y_predLogistique), 'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]), 'confusion': cm, 'precision': precision_score(self.y_test_lr, self.y_predLogistique), 'neg_precision':cm[1,1]/cm.sum(axis=1)[1], 'roc': roc_auc_score(self.y_test_lr, self.probs_lr)}
		# print(self.dictMetricLogistique)




	def regressionRandom(self):

		self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.X,self.Y)
		self.rm=RandomForestClassifier(n_estimators=500, bootstrap = True, max_features = 'sqrt')
		self.rm.fit(self.x_train,self.y_train)
		self.y_predRandom=self.rm.predict(self.x_test)
		self.confusion_rm=confusion_matrix(self.y_test,self.y_predRandom)
		self.probs_rm=self.rm.predict_proba(self.x_test)[:,1]
		cm=self.confusion_rm
		self.dictMetricRandom={'recall': recall_score(self.y_test, self.y_predRandom), 'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]), 'confusion': cm, 'precision': precision_score(self.y_test, self.y_predRandom), 'neg_precision':cm[1,1]/cm.sum(axis=1)[1], 'roc': roc_auc_score(self.y_test, self.probs_rm)}
		# print(self.dictMetricRandom)
		
		
	def regressionLineaire(self):
		
		self.indexNames = self.newdf[ self.newdf['Nwithdrawal_binary'] == 0 ].index
		self.dflinear=self.newdf
		self.dflinear.drop(self.indexNames , inplace=True)
		self.Ylinear = self.newdf.NW
		self.Xlinear = self.newdf >> select(X.lagged_NW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_NCG,X.SAS_NBP)
		self.x_train_lnr, self.x_test_lnr,self.y_train_lnr,self.y_test_lnr=train_test_split(self.Xlinear,self.Ylinear)
		self.lnr = LinearRegression()
		self.lnr.fit(self.x_train_lnr,self.y_train_lnr)
		self.y_predLinear=self.lnr.predict(self.x_test_lnr)
		#self.confusion_lnr=confusion_matrix(self.y_test_lnr,self.y_predLinear)
		#self.probs_lnr=self.lnr.predict_proba(self.x_test_lnr)[:,1]
		#cm=self.confusion_ln
		self.corr =pearsonr(self.y_test_lnr,self.y_predLinear)
		self.rmse=mean_squared_error(self.y_test_lnr, self.y_predLinear)
		self.average=np.average(self.y_test_lnr)
		self.anrmse=self.rmse/self.average
		self.min_max=max(self.y_test_lnr)-min(self.y_test_lnr) #je ne vois pas quelle colonne sélectionner...
		self.nrmse=self.rmse/self.min_max
		self.dictMetricLinear={'r2': r2_score(self.y_test_lnr, self.y_predLinear), 'rmse': self.rmse, 'nrmse': self.nrmse, 'anrmse': self.anrmse, 'cor': self.corr}
		# print(self.dictMetricLinear)





	def bestRegression(self):

		c1=0 #on définit un compteur pour la régression logistique
		c2=0 #on définit un compteur pour la random forest

		v1= self.dictMetricLogistique['recall']
		v2= self.dictMetricRandom['recall']
		if v1<v2:
			c2+=1
		else : 
			c1+=1

		v1= self.dictMetricLogistique['neg_recall']
		v2= self.dictMetricRandom['neg_recall']
		if v1<v2:
			c2+=1
		else : 
			c1+=1

		v1= self.dictMetricLogistique['precision']
		v2= self.dictMetricRandom['precision']
		if v1<v2:
			c2+=1
		else : 
			c1+=1

		v1= self.dictMetricLogistique['neg_precision']
		v2= self.dictMetricRandom['neg_precision']
		if v1<v2:
			c2+=1
		else : 
			c1+=1

		v1= self.dictMetricLogistique['roc']
		v2= self.dictMetricRandom['roc']
		if v1<v2:
			c2+=1
		else : 
			c1+=1

		#On compare les deux modèles
		if c1>=c2:
			return "logistique"
		else: 
			return "forest"
 
if __name__ == '__main__':
    # set_wd()
    dictionaire=import_xlsx()
    priceData =importPriceData()
    # print(dictionaire)
        
    fichier=fichierExcel(dictionaire)
    fichier.collectAllRegression(priceData)
    
