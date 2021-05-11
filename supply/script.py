
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



def set_wd(wd="/media/sf_flipo_partage/Documents/IN104/TB/FLIPO_GUILLAUME_GERARD_MATTHIEU_TB/supply"):
   os.chdir(wd)


#1) import consumption data from DE.csv into a pandas DataFrame and rename Date (CET) column to Date
#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_xlsx(f_name = "storage_data.xlsx", delimeter = ";"):
    data=pd.ExcelFile(f_name)
    return data

def importPriceData(fName='price_data.csv'):
	priceData=pd.read_csv(fName, sep=";")
	priceData=priceData.dropna() >> mutate(Date = pd.to_datetime(priceData['Date'], format = "%d/%m/%Y"))
	priceData.rename(columns={'Date' : 'gasDayStartedOn' }, inplace=True)
	return priceData


class fichierExcel:

	def __init__(self, fichier):
		self.allNameSheet=fichier
		self.dictionaire1=dict()
		self.dictionaire2=dict()
    # Import now the price_data.csv file it has 4 columns
# Change the name of the Date column to gasDayStartedon (just like the name in the storage_data.xlsx)
# The other columns represent the time spread columns
	
# 	def importPriceData(self,fName='price_data.csv'):
# 		self.priceData=pd.read_csv(fName, sep=";", parse_dates=["Date"])
# 		self.priceData.rename(columns={'Date' : 'gasDayStartedOn' }, inplace=True)    
	# def innerJoin(self,priceData):
	# 	self.listDf=[]
	# 	for i in range(len(self.allNameSheet)):
	# 		self.listDf.append(sheet(self.allNameSheet[i]))
	# 		sheet.createColumn(self.listDf[i],priceData)
	def collectAllRegression(self,priceData):
		self.listSheet=[]
		for i in range(len(self.allNameSheet)):
			self.listSheet.append(sheet(self.allNameSheet[i]))
			sheet.createColumn(self.listSheet[i],priceData)
			sheet.regressionLogistique(self.listSheet[i])
			sheet.regressionRandom(self.listSheet[i])





	#def bestOverAllregression(self):

	

class sheet:

	def __init__(self, dataFrameName):
		self.df=pd.read_excel(dictionaire,dataFrameName)
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

		self.Y = self.newdf >> select(X.Nwithdrawal_binary)
		self.X = self.newdf >> select(X.NW,X.lagged_NW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_NCG,X.SAS_NBP)
		self.Xarray = np.array(self.X)
		self.Yarray = np.array(self.Y['Nwithdrawal_binary'])
		self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.Xarray,self.Yarray,random_state=1)
		self.lr = LogisticRegression()
		self.lr.fit(self.x_train,self.y_train)
		self.y_predLogistique=self.lr.predict(self.x_test)
		# print(self.lr.coef_)
		# print(self.lr.intercept_)
		self.confusion_lr=confusion_matrix(self.y_test,self.y_predLogistique)
		# print(self.confusion)
		self.probs=self.lr.predict_proba(self.x_test)[:,1]
		cm=self.confusion_lr
		self.dictMetricLogistique={'recall': recall_score(self.y_test, self.y_predLogistique), 'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]), 'confusion': cm, 'precision': precision_score(self.y_test, self.y_predLogistique), 'neg_precision':cm[1,1]/cm.sum(axis=1)[1], 'roc': roc_auc_score(self.y_test, self.probs)}
		print(self.dictMetricLogistique)




	def regressionRandom(self):

		self.Y = self.newdf >> select(X.Nwithdrawal_binary)
		self.X = self.newdf >> select(X.NW,X.lagged_NW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_NCG,X.SAS_NBP)
		self.Xarray = np.array(self.X)
		self.Yarray = np.array(self.Y['Nwithdrawal_binary'])
		self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.Xarray,self.Yarray,random_state=1)
		self.rm=RandomForestClassifier()
		self.rm.fit(self.x_train,self.y_train)
		self.y_predRandom=self.rm.predict(self.x_test)
		self.confusion_rm=confusion_matrix(self.y_test,self.y_predRandom)
		self.probs=self.rm.predict_proba(self.x_test)[:,1]
		cm=self.confusion_rm
		self.dictMetricRandom={'recall': recall_score(self.y_test, self.y_predRandom), 'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]), 'confusion': cm, 'precision': precision_score(self.y_test, self.y_predRandom), 'neg_precision':cm[1,1]/cm.sum(axis=1)[1], 'roc': roc_auc_score(self.y_test, self.probs)}
		print(self.dictMetricRandom)





	#def bestRegression(self,)
 
if __name__ == '__main__':
    set_wd()
    plt.close()
    dictionaire=import_xlsx()
    priceData =importPriceData()
    # dfRehen=sheet(dictionaire.sheet_names[0])
    # df2=sheet(dictionaire.sheet_names[1])
    # sheet.createColumn(dfRehen,priceData)
    # sheet.createColumn(df2,priceData)
    
    fichier=fichierExcel(dictionaire.sheet_names)
    fichierExcel.collectAllRegression(fichier,priceData)
    # for i in range(len(dictionaire.sheet_names)):
    # 	print(fichier.allNameSheet[i], '\n',fichier.listDf[i].newdf.head(10))
    # sheet.regressionLogistique(dfRehen)
    # sheet.regressionRandom(dfRehen)
    # sheet.regressionLogistique(df2)
    # sheet.regressionRandom(df2)