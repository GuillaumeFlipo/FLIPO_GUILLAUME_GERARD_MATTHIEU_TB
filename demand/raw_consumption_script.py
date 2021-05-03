import pandas as pd
import os
from dfply import * # https://github.com/kieferk/dfply#rename
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#set working directory
def set_wd(wd="/media/sf_flipo_partage/Documents/IN104/TB/FLIPO_GUILLAUME_GERARD_MATTHIEU_TB/demand"):
   os.chdir(wd)


#1) import consumption data from DE.csv into a pandas DataFrame and rename Date (CET) column to Date
#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_csv(f_name = "DE.csv", delimeter = ";"):
    data=pd.read_csv("DE.csv", sep=";", parse_dates=["Date (CET)"])
    return data 

    # The LDZ represents gas consumption in GWh, Actual is the Actual temperature and Normal is the normal temperature
    
    # Try to use dfply pipes to rename

    
    # Plot using Matplotlib all three series on 3 sub plots to see them varying together
def plot_scatter(DataFrame):
    
    DataFrame.plot.scatter(x="Actual", y="LDZ", alpha=0.5, s=0.7)
    
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))
    # Do not forget to add a legend and a title to the plot

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
    
    # Comment on their variation and their relationships


    
    # use dfply to transform Date column to DateTime type

    

#2) work on consumption data (non-linear regression)
#2)1. Plot with a scatter plot the consumption as a function of temperature


#2)2. define the consumption function (I give it to you since it is hard to know it without experience)
'''
#This function takes temperature and 4 other curve shape parameters and returns a value of consumption
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))

#These are random initial values of a, b, c, d
guess_values= [500, -25, 2, 100]

#2)3. Fill out this h_hat array with values from the function h

# You will take the 'Actual' column from the DE.csv file as being the input temperature so its length should be the number of rows in the DataFrame imported
h_hat = np.empty(len())'''


    
    
# For each value of temmperature of this column you will calculate the consumption using the h function above
# DO NOT use a for loop, vectorize
# Use the array guess_values for the curve parameters a, b, c, d that is to say a = guess_values[0], b = guess_values[1], c = guess_values[2], d = guess_values[3]

# Plot on a graph the real consumption (LDZ column) as a function of Actual temperature use blue dots
# On this same graph add the h_hat values as a function of Actual temperature use a red line for this
# Do not forget to add a legend and a title to the plot
# Play around with the parameters in guess_values until you feel like your curve is more or less correct


#2)4. optimize the parameters

    # Your goal right now is to find the optimal values of a, b, c, d using SciPy
    # Inspire yourselves from the following video
    # https://www.youtube.com/watch?v=4vryPwLtjIY

#2)5. check the new fit

#Repeat what we did in 2)3. but with the new optimized coefficients a, b, c, d


#calculate goodness of fit parameters: correlation, root mean square error (RMSE), Average normalised RMSE, normalized RMSE
#averaged normalized RMSE is RMSE/(average value of real consumption)
#normalized RMSE is RMSE/(max value of real consumption - min value of real consumption)
#Any other metric we could use ?

if __name__ == '__main__':
    set_wd()
    data=import_csv()
    plot_scatter(data)
    #data.plot.scatter(x="Actual", y="LDZ", alpha=0.5, s=0.7)
    #print(data.head())
    plt.show()