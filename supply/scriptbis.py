import pandas as pd
import os
from dfply import * # https://github.com/kieferk/dfply#rename
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def set_wd(wd="/media/sf_flipo_partage/Documents/IN104/TB/FLIPO_GUILLAUME_GERARD_MATTHIEU_TB/supply"):
   os.chdir(wd)


#1) import consumption data from DE.csv into a pandas DataFrame and rename Date (CET) column to Date
#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_xlsx(f_name = "storage_data.xlsx", delimeter = ";"):
    data=pd.ExcelFile(f_name)
    return data


class fichierExcel:

	def __init__(self, fichier,dictionnaire):
		self.allNameSheet=fichier
		self.dictionnaire=dictionnaire
		# self.dictionnaire1={}
		# self.dictionnaire2={}
    # Import now the price_data.csv file it has 4 columns
# Change the name of the Date column to gasDayStartedon (just like the name in the storage_data.xlsx)
# The other columns represent the time spread columns
	
	def importPriceData(self,fName='price_data.csv'):
		self.priceData=pd.read_csv(fName, sep=";", parse_dates=["Date"])
		self.priceData.rename(columns={'Date' : 'gasDayStartedOn' }, inplace=True)
        

	def innerJoin(self):
		self.listDf=list()
		for i in range(len(self.allNameSheet)):

			self.listDf.append(sheet(self.allNameSheet[i]))
			self.listDf[i]= self.listDf[i].createColumn()
			

			

		



    #def collectAllRegression(self):

	#def bestOverAllregression(self):

	

class sheet:

	def __init__(self, dataFrameName):
		self.df=pd.read_excel(dictionnaire,dataFrameName)
		self.len=len(self.df.full)

	def createColumn(self):
		
		self.newdf=self.df >> select(X.gasDayStartedOn,X.full,X.injection,X.withdrawal)
		self.newdf['NW']=self.newdf['withdrawal']-self.newdf['injection']
		self.newdf['lagged_NW']=self.newdf.NW.shift(1)
		self.newdf['Nwithdrawal_binary'] = np.where(self.newdf['NW']>0, 1, 0)
		self.newdf['FSW1']=np.where((self.newdf['full']-45)>0,self.newdf['full']-45,0)
		self.newdf['FSW2']=np.where((45-self.newdf['full'])>0,45-self.newdf['full'],0)
		self.newdf=self.newdf >> select(X.gasDayStartedOn,X.NW,X.lagged_NW,X.Nwithdrawal_binary,X.FSW1,X.FSW2)
		self.newdf= 


	#def regressionLogistique(self,):

	#def regressionRandom(self,):

	#def bestRegression(self,)
 
if __name__ == '__main__':
    set_wd()
    plt.close()
    dictionnaire=import_xlsx()
    # print(type(dictionnaire))
    # dictionnaireBis=fichierExcel()
    dfRehen=sheet(dictionnaire.sheet_names[0])
    sheet.createColumn(dfRehen)
    fichier=fichierExcel(dictionnaire.sheet_names)
    fichier.innerJoin()

    print(fichier.dictionnaire1)
    # print(dfRehen.newdf.head())
