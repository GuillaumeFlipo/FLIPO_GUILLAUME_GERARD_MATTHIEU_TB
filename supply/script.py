
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
import openpyxl



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
   		else :
   			print("Best regression is LogisticRegression")

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
		self.corrlr,self.cov =pearsonr(self.y_test_lr,self.y_predLogistique)
		self.rmselr=mean_squared_error(self.y_test_lr, self.y_predLogistique)
		self.average=np.average(self.y_test_lr)
		self.armselr=self.rmselr/self.average
		self.min_max=max(self.y_test_lr)-min(self.y_test_lr) #je ne vois pas quelle colonne sélectionner...
		self.nrmselr=self.rmselr/self.min_max
		self.dictMetricLogistiqueBis={'r2': r2_score(self.y_test_lr, self.y_predLogistique), 'rmselr': self.rmselr, 'nrmselr': self.nrmselr, 'armselr': self.armselr, 'cor': self.corrlr}




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
		# print(self.lnr.coef_)
		self.corrlnr,self.cov =pearsonr(self.y_test_lnr,self.y_predLinear)
		self.rmselnr=mean_squared_error(self.y_test_lnr, self.y_predLinear)
		self.average=np.average(self.y_test_lnr)
		self.armselnr=self.rmselnr/self.average
		self.min_max=max(self.y_test_lnr)-min(self.y_test_lnr) #je ne vois pas quelle colonne sélectionner...
		self.nrmselnr=self.rmselnr/self.min_max
		self.dictMetricLinear={'r2': r2_score(self.y_test_lnr, self.y_predLinear), 'rmselnr': self.rmselnr, 'nrmselnr': self.nrmselnr, 'armselnr': self.armselnr, 'cor': self.corrlnr}
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



class requirement:
    def __init__(self,fichier):
        self.dictLogistique=dict()
        self.dictMultilineaire=dict()
        self.dictCoefLog=dict()
        self.dictCoefMulti=dict()
        self.fichier=fichier

    # def createDictLogistiqueMetrics(self):
    # 	self.dictLogistique['recall']= []
    # 	self.dictLogistique['neg_recall']= [] 
    # 	self.dictLogistique['precision']= []
    # 	self.dictLogistique['neg_precision']= []
    # 	self.dictLogistique['roc']= []

    # 	for i in range(len(self.fichier.allNameSheet)):
    # 		self.dictLogistique['recall'].append(self.fichier.listSheet[i].dictMetricLogistique['recall']) 		
    # 		self.dictLogistique['neg_recall'].append(self.fichier.listSheet[i].dictMetricLogistique['neg_recall'])
    # 		self.dictLogistique['precision'].append(self.fichier.listSheet[i].dictMetricLogistique['precision'])
    # 		self.dictLogistique['neg_precision'].append(self.fichier.listSheet[i].dictMetricLogistique['neg_precision'])
    # 		self.dictLogistique['roc'].append(self.fichier.listSheet[i].dictMetricLogistique['roc'])
    # 	self.dfLog=pd.DataFrame(self.dictLogistique)
    # 	print(self.dfLog)

    def createDictLogistiqueMetrics(self):
    	self.dictLogistique= {'Logistique':[],'corr': [],'rmse': [],'nrmse': [],'armse': []}
    	for i in range(len(self.fichier.allNameSheet)):
    		self.dictLogistique['corr'].append(self.fichier.listSheet[i].dictMetricLogistiqueBis['cor']) 		
    		self.dictLogistique['rmse'].append(self.fichier.listSheet[i].dictMetricLogistiqueBis['rmselr'])
    		self.dictLogistique['nrmse'].append(self.fichier.listSheet[i].dictMetricLogistiqueBis['nrmselr'])
    		self.dictLogistique['armse'].append(self.fichier.listSheet[i].dictMetricLogistiqueBis['armselr'])
    		self.dictLogistique['Logistique'].append(self.fichier.listSheet[i].dfname)
    	self.dfLog=pd.DataFrame(self.dictLogistique)
    	print(self.dfLog)


        
        

    def createDictMultilineaireMetrics(self):
    	self.dictMultilineaire= {'Multilineaire':[],'corr': [],'rmse': [],'nrmse': [],'armse': []}
    	for i in range(len(self.fichier.allNameSheet)):
    		self.dictMultilineaire['corr'].append(self.fichier.listSheet[i].dictMetricLinear['cor']) 		
    		self.dictMultilineaire['rmse'].append(self.fichier.listSheet[i].dictMetricLinear['rmselnr'])
    		self.dictMultilineaire['nrmse'].append(self.fichier.listSheet[i].dictMetricLinear['nrmselnr'])
    		self.dictMultilineaire['armse'].append(self.fichier.listSheet[i].dictMetricLinear['armselnr'])
    		self.dictMultilineaire['Multilineaire'].append(self.fichier.listSheet[i].dfname)
    	self.dfMulti=pd.DataFrame(self.dictMultilineaire)
    	print(self.dfMulti)

    def createDictCoefLog(self):
    	self.dictCoefLog={'Logistique':[],'lagged_NW': [], 'FSW1':[],'FSW2':[],'SAS_GPL':[],'SAS_NCG':[],'SAS_NBP':[]}
    	for i in range(len(self.fichier.allNameSheet)):
    		self.dictCoefLog['lagged_NW'].append(self.fichier.listSheet[i].lr.coef_[0][0])
    		self.dictCoefLog['FSW1'].append(self.fichier.listSheet[i].lr.coef_[0][1])
    		self.dictCoefLog['FSW2'].append(self.fichier.listSheet[i].lr.coef_[0][2])
    		self.dictCoefLog['SAS_GPL'].append(self.fichier.listSheet[i].lr.coef_[0][3])
    		self.dictCoefLog['SAS_NCG'].append(self.fichier.listSheet[i].lr.coef_[0][4])
    		self.dictCoefLog['SAS_NBP'].append(self.fichier.listSheet[i].lr.coef_[0][5])
    		self.dictCoefLog['Logistique'].append(self.fichier.listSheet[i].dfname)
    	self.dfCoefLog=pd.DataFrame(self.dictCoefLog)
    	print(self.dfCoefLog)

    def createDictCoefMulti(self):
    	self.dictCoefMulti={'Multilineaire':[],'lagged_NW': [], 'FSW1':[],'FSW2':[],'SAS_GPL':[],'SAS_NCG':[],'SAS_NBP':[]}
    	for i in range(len(self.fichier.allNameSheet)):
    		self.dictCoefMulti['lagged_NW'].append(self.fichier.listSheet[i].lnr.coef_[0])
    		self.dictCoefMulti['FSW1'].append(self.fichier.listSheet[i].lnr.coef_[1])
    		self.dictCoefMulti['FSW2'].append(self.fichier.listSheet[i].lnr.coef_[2])
    		self.dictCoefMulti['SAS_GPL'].append(self.fichier.listSheet[i].lnr.coef_[3])
    		self.dictCoefMulti['SAS_NCG'].append(self.fichier.listSheet[i].lnr.coef_[4])
    		self.dictCoefMulti['SAS_NBP'].append(self.fichier.listSheet[i].lnr.coef_[5])
    		self.dictCoefMulti['Multilineaire'].append(self.fichier.listSheet[i].dfname)
    	self.dfCoefMulti=pd.DataFrame(self.dictCoefMulti)
    	print(self.dfCoefMulti)

    def createExcel(self):
    	self.createDictLogistiqueMetrics()
    	self.createDictMultilineaireMetrics()
    	self.createDictCoefLog()
    	self.createDictCoefMulti()
    	writer=pd.ExcelWriter('../demand/demand.xlsx', mode='a',engine='openpyxl')
    	self.dfLog.to_excel(writer,index=False,sheet_name="supply")
    	self.dfMulti.to_excel(writer,index=False,sheet_name="supply",startrow=len(self.dfLog)+2)
    	self.dfCoefLog.to_excel(writer,index=False,sheet_name="supply",startcol=6)
    	self.dfCoefMulti.to_excel(writer,index=False,sheet_name="supply",startcol=6,startrow=len(self.dfLog)+2)
    	writer.save()








 
if __name__ == '__main__':
    # set_wd()
    dictionaire=import_xlsx()
    priceData =importPriceData()
    # print(dictionaire)
        
    fichier=fichierExcel(dictionaire)
    fichier.collectAllRegression(priceData)
    solution=requirement(fichier)
    solution.createExcel()
    

# self.fichier.listSheet[i].dictMetricLinear['cor']) 
   
# self.fichier.listSheet[i].dictMetricLinear['rmselnr']) 
# self.fichier.listSheet[i].dictMetricLinear['nrmselnr'])
# self.fichier.listSheet[i].dictMetricLinear['armselnr'])