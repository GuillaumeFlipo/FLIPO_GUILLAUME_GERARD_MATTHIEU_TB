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

#This function sets the working directory
def set_wd(wd="/media/sf_flipo_partage/Documents/IN104/TB/FLIPO_GUILLAUME_GERARD_MATTHIEU_TB/demand"):
   os.chdir(wd)

#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_csv(f_name = "DE.csv", delimeter = ";"):
    f = data=pd.read_csv("DE.csv", sep=";", parse_dates=["Date (CET)"])
    f.rename(columns={'Date (CET)' : 'Date' }, inplace=True)
    return f.dropna()

#This function creates a scatter plot given a DataFrame and an x and y column
def scatter_plot(dataframe):
    plt.scatter(dataframe['Actual'],dataframe['LDZ'],color='blue',alpha=0.5,s=0.7,label='LDZ zn fonction de Actual')
    plt.legend()
    plt.show()
    

#This function is the sigmoid function for gas consumption as a function of temperature
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))


#The following function takes sigmoid parameters values and a temperature vector as input and plots the sigmoid, can compare it with the actual values
def consumption_sigmoid(t, real_conso, a = 900, b = -35, c = 6, d = 300, plot = True):
  
    h_hat = h(t, a, b, c, d)

    if plot== True:
        plt.plot(t,h_hat,color='r',label='théorique')
        plt.xlabel('temperature (C°)')
        plt.ylabel('consommation')
        plt.plot()
        #if real_conso is not None you plot it as well
        if not isinstance(real_conso, type(None)):
            plt.scatter(real_conso.Actual,real_conso.LDZ,color='blue',alpha=0.5,s=0.7,label='réelle')
            plt.title('Comparaison entre la consomation réelle et la sigmoide théorique')
            if(len(t) != len(real_conso)):
                print("Difference in length between Temperature and Real Consumption vectors")
            # add title and legend and show plot
        plt.legend()
        plt.show()
    return h_hat

#The following function gets the fit metrics list between 2 sigmoids
def get_fit_metrics(h_hat, real_conso):
    if(len(h_hat) != len(real_conso)):
        print("Difference in length between Fit and Real Consumption vectors")
    else:
        y_true=real_conso.values
        y_pred=h_hat
        corr = r2_score(y_true,y_pred)
        rmse=mean_squared_error(y_true,y_pred)
        average_y_true=np.average(y_true)
        min_y_true=max(y_true)-min(y_true)
        anrmse=rmse/average_y_true
        nrmse=rmse/min_y_true

    return [corr, rmse, nrmse, anrmse]

#The following class is the cosumption class it takes sigmoid parameters as well as a temperature as input
class consumption:
    #Initialize class
    def __init__(self, a, b, c, d):
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        

    #calculate the consumption given a temperature
    def get_consumption(self, temperature):
        self.conso=h(temperature,self.a , self.b , self.c, self.d )

        

    #get the sigmoid considering a temperature between -40 and 39, use the function consumption_sigmoid above
    def sigmoid(self, p):
        self.sigmoidConso=consumption_sigmoid(np.linspace(-40,39,1000),None, self.a , self.b , self.c, self.d , plot = p)


    #This is what the class print if you use the print function on it
    def __str__(self):
        
        return 
#The following class optimizes the parameters of the sigmoid and returns an object of class consumption
class optimize_sigmoid:
    #Initialize guess values that are common to all instances of the clasee
    # __guess_a, __guess_b, __guess_c, __guess_d

    def __init__(self, f):
        if isinstance(f, pd.DataFrame):
            if 'Actual' and 'LDZ' in f.columns:
                self.__f=f
                self.guess=[500, -25, 2, 100]
                self.t = np.linspace(min(self.__f['Actual'].values),max(self.__f['Actual'].values),len(self.__f.Actual))

                
            else:
                print("Class not initialized since f does not contain Actual and LDZ column names")
        else:
            print("Class not initialized since f is not a DataFrame")

    #optimize and return metrics use functions h, consumption_sigmoid defined above as well as get_fit_metrics
    def optimize(self):
        if self.__f is not None:
            self.__coef, self.__cov = curve_fit(h,self.__f['Actual'].values,self.__f['LDZ'].values,self.guess)
            
            s = consumption_sigmoid(self.t, self.__f,self.__coef[0], self.__coef[1], self.__coef[2], self.__coef[3],True)
            
            self.__corr, self.__rmse, self.__nrmse, self.__anrmse = get_fit_metrics(s, self.__f['LDZ'])
        else:
            print("Class not initialized")

    #this function returns the fit metrics calculated above
    def fit_metrics(self):
        if  self.__corr is not None:
            return [self.__corr, self.__rmse, self.__nrmse, self.__anrmse]
        else:
            print("optimize method is not yet run")

    #This function creates the class consumption
    def create_consumption(self):
        if self.__f is not None:
            return consumption(self.__coef[0], self.__coef[1], self.__coef[2], self.__coef[3])
        else:
            print("optimize method is not yet run")

    #This is what the class print if you use the print function on it
    def __str__(self):
        if self.__f is not None:
            t = "optimize method has ran"
        else:
            t = "optimize method is not yet run"
        return t

    def getCoeff(self):
        if self.__coef is not None:
            return self.__coef

#If you have filled correctly the following code will run without an issue        
class requirement:
    def __init__(self,sig):
        self.dict=dict()
        self.optimizeSigmoidObject=sig

    def createDict(self):
        self.dict={'A' : [self.optimizeSigmoidObject.getCoeff()[0]],'B' : [self.optimizeSigmoidObject.getCoeff()[1]],'C' : [self.optimizeSigmoidObject.getCoeff()[2]],'D' : [self.optimizeSigmoidObject.getCoeff()[3]]}
        self.dict['corr']= [self.optimizeSigmoidObject.fit_metrics()[0]]
        self.dict['rmse']= [self.optimizeSigmoidObject.fit_metrics()[1]]
        self.dict['nrmse']= [self.optimizeSigmoidObject.fit_metrics()[2]]
        self.dict['armse']= [self.optimizeSigmoidObject.fit_metrics()[3]]
        self.dfrequirement=pd.DataFrame(data=self.dict)
        # print(self.dfrequirement)
        self.dfrequirement.to_excel('Demand.xlsx')
        return self.dfrequirement

if __name__ == '__main__':

    #set working directory
    set_wd()

    #1) import consumption data and plot it
    conso = import_csv()
    print(conso.head())


    #2) work on consumption data (non-linear regression)
    #2)1. Plot consumption as a function of temperature    

    scatter_plot(conso)        

    #2)2. optimize the parameters
    sig = optimize_sigmoid(conso)
    sig.optimize()
    # print(sig.fit_metrics())
    c = sig.create_consumption()
    print(sig)
    print(sig.getCoeff())
    print(sig.guess)


    # #2)3. check the new fit

    # # These are the 3 ways to access a protected attribute, it works the same for a protected method
    # # An attribute/method is protected when it starts with 2 underscores "__"
    # # Protection is good to not falsy create change
    
    print(
            [
            sig.__dict__['_optimize_sigmoid__corr'],
            sig.__dict__['_optimize_sigmoid__rmse'],
            sig.__dict__['_optimize_sigmoid__nrmse'],
            sig.__dict__['_optimize_sigmoid__anrmse']
            ]
        )

    print(
            [
            sig._optimize_sigmoid__corr,
            sig._optimize_sigmoid__rmse,
            sig._optimize_sigmoid__nrmse,
            sig._optimize_sigmoid__anrmse
            ]
        )

    print(
            [
            getattr(sig, "_optimize_sigmoid__corr"),
            getattr(sig, "_optimize_sigmoid__rmse"),
            getattr(sig, "_optimize_sigmoid__nrmse"),
            getattr(sig, "_optimize_sigmoid__anrmse")
            ]
        )
    
    print(sig.fit_metrics())
    c.sigmoid(True)
    solution = requirement(sig)
    dfrequirement=solution.createDict()
    
    #3) If time allows do TSA on actual temperature
    #3)1. Check trend (and Remove it)
    #3)2. Check Seasonality (Normal Temperature)
    #3)3. Model stochastic part that is left with ARIMA
    #3)4. Use this to forecast consumption over N days
