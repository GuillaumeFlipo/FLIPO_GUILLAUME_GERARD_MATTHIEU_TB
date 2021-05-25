
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
import consumption_script as cs



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
      self.listSheet_Bl=[]
      

   	# def bestRegressionOverAll(self):
   	# 	self.compteurRandom = 0;
   	# 	self.compteurLogistique = 0;
   	# 	for i in self.allNameSheet:

   def createDfDemand(self):
      conso = cs.import_csv()
      sig = cs.optimize_sigmoid(conso)
      sig.optimize(False)
      self.consoCoef=sig.getCoeff()
      self.Dfdemand=pd.DataFrame(columns=['gasDayStartedOn','Conso_pred'])
      self.Dfdemand['gasDayStartedOn']=conso['Date']
      self.consoModel=cs.h(np.array(conso['Actual'].values),self.consoCoef[0],self.consoCoef[1],self.consoCoef[2],self.consoCoef[3])
      self.Dfdemand['Conso_pred']=self.consoModel
      self.Dfdemand['LDZ']=conso['LDZ']
      


   def Balance(self):
   	  self.AllSheet=pd.DataFrame(columns=['gasDayStartedOn'])
   	  self.AllSheet_NW=pd.DataFrame(columns=['gasDayStartedOn'])
   	  for i in self.allNameSheet:
   	  	self.AllSheet['gasDayStartedOn']=self.d[i]['gasDayStartedOn']
   	  	self.AllSheet_NW['gasDayStartedOn']=self.d[i]['gasDayStartedOn']
   	  	break

   	  
   	  for i in self.allNameSheet:
   	  	f = self.d[i]
   	  	stBalance = sheet(i, f)
   	  	stBalance.createColumn(priceData)
   	  	stBalance.regressionLogistique()
   	  	stBalance.regressionLineaire()
   	  	stBalance.regressionfinale()
   	  	self.listSheet_Bl.append(stBalance)
   	  self.AllSheet = self.AllSheet.fillna(0)
   	  self.AllSheet_NW = self.AllSheet_NW.fillna(0)
   	  self.AllSheetSum_NW=pd.DataFrame(columns=['gasDayStartedOn'])
   	  self.AllSheetSum_NW['NW_reel']=self.AllSheet_NW.sum(axis=1,numeric_only=True)
   	  self.AllSheetSum_NW['gasDayStartedOn']= self.AllSheet['gasDayStartedOn']
   	  self.AllSheetSum=pd.DataFrame(columns=['gasDayStartedOn'])
   	  self.AllSheetSum['Supply_pred']=self.AllSheet.sum(axis=1,numeric_only=True)
   	  self.AllSheetSum['gasDayStartedOn']= self.AllSheet['gasDayStartedOn']
   	  self.AllSheetSum['NW_reel']=self.AllSheetSum_NW['NW_reel']
   	  self.indexNames = self.AllSheetSum[ self.AllSheetSum['Supply_pred'] < 1 ].index
   	  self.AllSheetSum.drop(self.indexNames , inplace=True)
   	  self.AllSheetFinal = pd.merge(self.AllSheetSum,self.Dfdemand, on="gasDayStartedOn")
   	  # self.AllSheetFinal['prediction']=np.where((self.AllSheetFinal['Supply_pred']-self.AllSheetFinal['Conso_pred'])>0,"Buy","Sell")
   	  # self.AllSheetFinal['Vrai_Data']=np.where((self.AllSheetFinal['NW_reel']-self.AllSheetFinal['LDZ'])>0,"Buy","Sell")
   	  self.AllSheetFinal['prediction']=self.AllSheetFinal['Supply_pred']-self.AllSheetFinal['Conso_pred']
   	  self.AllSheetFinal['prediction']=self.AllSheetFinal['prediction'].map(self.fonction_decision)
   	  self.AllSheetFinal['Vrai_Data'] = self.AllSheetFinal['NW_reel']-self.AllSheetFinal['LDZ']
   	  self.AllSheetFinal['Vrai_Data']=self.AllSheetFinal['Vrai_Data'].map(self.fonction_decision)
   	  self.AllSheetFinal['result']=np.where(self.AllSheetFinal['prediction']==self.AllSheetFinal['Vrai_Data'],1,0)
   	  self.Sum=self.AllSheetFinal['result'].sum()
   	  print(self.AllSheetFinal.head())
   	  print("Nombre de bons résultats : ",self.Sum )
   	  print("Nombre total Possible de bons résultats : ",len(self.AllSheetFinal['result'].values))
   	  self.precision=self.Sum/len(self.AllSheetFinal['result'].values)
   	  self.Sum45=self.AllSheetFinal.head(25)['result'].sum()
   	  self.precision45=self.Sum45/25
   	  print("La précision est de : ",self.precision)


   def fonction_decision(self,value):
   	if (value>0):
   		return "Buy"
   	elif (value<0):
   		return "Sell"
   	else :
   		return "Flat"

        




   		
   		

	

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
		self.Datedf =self.newdf >> select(X.gasDayStartedOn,X.NW,X.lagged_NW,X.Nwithdrawal_binary,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_NCG,X.SAS_NBP)
		
		


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
		# print(self.dictMetricLinear
		


	def regressionfinale(self):
		
		self.y_pred_Bin_Bl=self.lr.predict(self.X)
		self.dict_y_pred_binary={ 'y_pred_binary' : self.y_pred_Bin_Bl, 'gasDayStartedOn': self.Datedf['gasDayStartedOn'].values}
		self.df_y_pred_binary=pd.DataFrame(self.dict_y_pred_binary)
		self.dfBalance=pd.merge(self.Datedf,self.df_y_pred_binary, on='gasDayStartedOn')
		# self.dfBalance=pd.concat(self.newdf,self.df_y_pred_binary,axis=1)
		self.dfBalance=self.dfBalance >> select(X.lagged_NW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_NCG,X.SAS_NBP,X.gasDayStartedOn,X.y_pred_binary)
		self.temp=self.dfBalance >> mask(self.df_y_pred_binary['y_pred_binary']>0)
		self.X_reg=self.temp >> select(X.lagged_NW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_NCG,X.SAS_NBP)
		self.y_pred_num_Bl=self.lnr.predict(self.X_reg)
		# self.x_train_lnr, self.x_test_lnr,self.y_train_lnr,self.y_test_lnr=train_test_split(self.XlinearBalance,self.YlinearBalance)
		# self.y_pred_num=self.lnrBalance.predict(self.X)
		self.dict_y_pred_num={ 'y_pred_num' : self.y_pred_num_Bl, 'gasDayStartedOn': self.temp['gasDayStartedOn'].values}
		self.df_y_pred_num=pd.DataFrame(self.dict_y_pred_num)
		self.temp=pd.merge(self.temp,self.df_y_pred_num, on='gasDayStartedOn')
		self.dfBalance = pd.merge(self.dfBalance, self.temp, on = "gasDayStartedOn", how  = "outer")
		self.dfBalance = self.dfBalance.fillna(0)
		self.final_df =  self.dfBalance >> select(X.gasDayStartedOn,X.y_pred_num)
		self.final_df2 = self.Datedf >> select(X.NW,X.gasDayStartedOn)
		# self.final_df3 = pd.merge(self.final_df,self.final_df2,on = "gasDayStartedOn", how  = "outer")
		fichier.AllSheet_NW = pd.merge(fichier.AllSheet_NW,self.final_df2,on = "gasDayStartedOn", how  = "outer")
		fichier.AllSheet = pd.merge(fichier.AllSheet,self.final_df,on = "gasDayStartedOn", how  = "outer")
		



    



class requirement:
    def __init__(self,fichier):
        self.fichier=fichier

    def createExcel(self):
    	self.dfFinal=self.fichier.AllSheetFinal
    	self.DictValues={'Nombre de jours': [1843,45],'Nbre_bon_res' : [self.fichier.Sum,self.fichier.Sum45],'Nbre_bon_res_possible': [len(self.fichier.AllSheetFinal['result'].values),25], 'Precision' : [self.fichier.precision,self.fichier.precision45] }
    	self.DfValues = pd.DataFrame(self.DictValues)
    	writer=pd.ExcelWriter('../demand/TB_IN104.xlsx', mode='a',engine='openpyxl')
    	self.dfFinal.to_excel(writer,index=False,sheet_name="balance")
    	self.DfValues.to_excel(writer,index=False,sheet_name="balance",startcol=10)
    	writer.save()








 
if __name__ == '__main__':
    # set_wd()
    dictionaire=import_xlsx()
    priceData =importPriceData()
    # print(dictionaire)
        
    fichier=fichierExcel(dictionaire)
    # fichier.collectAllRegression(priceData)
   
    fichier.createDfDemand()
    fichier.Balance()

    solution=requirement(fichier)
    solution.createExcel()
    

# self.fichier.listSheet[i].dictMetricLinear['cor']) 
   
# self.fichier.listSheet[i].dictMetricLinear['rmselnr']) 
# self.fichier.listSheet[i].dictMetricLinear['nrmselnr'])
# self.fichier.listSheet[i].dictMetricLinear['armselnr'])