import pandas as pd
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#This function sets the working directory
def set_wd(wd):
    os.chdir(wd)

#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_csv(f_name = "DE.csv", delimeter = ";", plot = True):

    return f >> mutate(Date = pd.to_datetime(conso['Date']))

#This function creates a scatter plot given a DataFrame and an x and y column
def scatter_plot(dataframe = "conso", x = "Actual", y = "LDZ", col = "red"):

#This function is the sigmoid function for gas consumption as a function of temperature
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))

#The following function takes sigmoid parameters values and a temperature vector as input and plots the sigmoid, can compare it with the actual values
def consumption_sigmoid(t, real_conso, a = 900, b = -35, c = 6, d = 300, plot = True):
  
    h_hat = h(t, a, b, c, d)

    if plot:
        plt.plot()
        #if real_conso is not None you plot it as well
        if not isinstance(real_conso, type(None)):
            plt.plot()
            if(len(t) != len(real_conso)):
                print("Difference in length between Temperature and Real Consumption vectors")
            # add title and legend and show plot
    return h_hat

#The following function gets the fit metrics list between 2 sigmoids
def get_fit_metrics(h_hat, real_conso):
    if(len(h_hat) != len(real_conso)):
        print("Difference in length between Fit and Real Consumption vectors")
    else:
        
    return []

#The following class is the cosumption class it takes sigmoid parameters as well as a temperature as input
class consumption:
    #Initialize class
    def __init__(self, a, b, c, d):
        

    #calculate the consumption given a temperature
    def get_consumption(self, temperature):
        

    #get the sigmoid considering a temperature between -40 and 39, use the function consumption_sigmoid above
    def sigmoid(self, p):

    #This is what the class print if you use the print function on it
    def __str__(self):
        
        return t

#The following class optimizes the parameters of the sigmoid and returns an object of class consumption
class optimize_sigmoid:
    #Initialize guess values that are common to all instances of the clasee
    __guess_a, __guess_b, __guess_c, __guess_d

    def __init__(self, f):
        if isinstance(f, pd.DataFrame):
            if 'Actual' and 'LDZ' in f.columns:
                
            else:
                print("Class not initialized since f does not contain Actual and LDZ column names")
        else:
            print("Class not initialized since f is not a DataFrame")

    #optimize and return metrics use functions h, consumption_sigmoid defined above as well as get_fit_metrics
    def optimize(self):
        if self.__f is not None:
            self.__coef, self.__cov = curve_fit(
                h,

                )
            
            s = consumption_sigmoid(

                plot = True
                )
            
            self.__corr, self.__rmse, self.__nrmse, self.__anrmse = get_fit_metrics(s, self.__f['LDZ'])
        else:
            print("Class not initialized")

    #this function returns the fit metrics calculated above
    def fit_metrics(self):
        if  is not None:
            return 
        else:
            print("optimize method is not yet run")

    #This function creates the class consumption
    def create_consumption(self):
        if  is not None:
            return 
        else:
            print("optimize method is not yet run")

    #This is what the class print if you use the print function on it
    def __str__(self):
        if is not None:

        else:
            t = "optimize method is not yet run"
        return t

#If you have filled correctly the following code will run without an issue        
if __name__ == '__main__':

    #set working directory
    set_wd()

    #1) import consumption data and plot it
    conso = import_csv()

    #2) work on consumption data (non-linear regression)
    #2)1. Plot consumption as a function of temperature    

    scatter_plot()        

    #2)2. optimize the parameters
    sig = optimize_sigmoid(conso)
    sig.optimize()
    c = sig.create_consumption()
    print(sig)


    #2)3. check the new fit

    # These are the 3 ways to access a protected attribute, it works the same for a protected method
    # An attribute/method is protected when it starts with 2 underscores "__"
    # Protection is good to not falsy create change
    
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
    print(c)
    
    #3) If time allows do TSA on actual temperature
    #3)1. Check trend (and Remove it)
    #3)2. Check Seasonality (Normal Temperature)
    #3)3. Model stochastic part that is left with ARIMA
    #3)4. Use this to forecast consumption over N days
