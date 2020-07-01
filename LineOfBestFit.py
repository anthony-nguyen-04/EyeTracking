import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

###################
# Author: Anthony Nguyen
# Date: ??? - 6/30/20
# Purpose:
#      Calculates where you are looking at on a screen via eye-tracking and machine learning
#
###################

# the percentage axis of the position-percentage graphs will always be constant (intervals of eighths)
percentArray = np.array([[0.00], [0.125], [0.250], [0.375], [0.500], [0.625], [0.750], [0.875], [1.000]])

#setting up a method for linear regression
def linearRegression(dependentAxis, independentAxis):

    #sets up a linear regression system and fits it with the input coordinates
    linearRegressionLine = LinearRegression()
    linearRegressionLine.fit(dependentAxis, independentAxis)

    #clears the graph before preparing to load in new ones
    plt.clf()

    #graphs the pre-existing coordinates,
    #  as well as a line for the machine-learning linear system prediction

    plt.scatter(dependentAxis, independentAxis, color='blue')
    plt.plot(dependentAxis, linearRegressionLine.predict(dependentAxis), color='red')

    #labelling the graph
    plt.title('Linear Regression')
    plt.xlabel('Position')
    plt.ylabel('Percent')
    plt.xticks(np.arange(100.00, 250.00, step=25.00))
    plt.yticks(np.arange(0.00, 1.00, step=.10))

    #print(linearRegressionLine.predict([[7]]))

    plt.show()

    return linearRegressionLine

def polynomialRegression(dependentAxis, independentAxis, power):
    if power < 1 or power > 5:
        polynomialRegression(dependentAxis, independentAxis, 2)
    else:

        #sets up a polynomial regression system and fits it with the input coordinates
        poly = PolynomialFeatures(degree=power)
        dependentPolynomial = poly.fit_transform(dependentAxis)
        poly.fit(dependentPolynomial, independentAxis)

        polynomialRegressionLine = LinearRegression()
        polynomialRegressionLine.fit(dependentPolynomial, independentAxis)

        # clears the graph before preparing to load in new ones
        plt.clf()

        # graphs the pre-existing coordinates,
        #  as well as a line for the machine-learning polynomial system prediction
        plt.scatter(dependentAxis, independentAxis, color='blue')
        plt.plot(dependentAxis, polynomialRegressionLine.predict(poly.fit_transform(dependentAxis)), color='red')

        # labelling the graph
        plt.title('Polynomial Regression')
        plt.xlabel('Position')
        plt.ylabel('Percent')
        plt.xticks(np.arange(100.00, 250.00, step=25.00))
        plt.yticks(np.arange(0.00, 1.00, step=.10))

        #print(polynomialRegressionLine.predict(poly.fit_transform([[7]])))

        plt.show()

        return (polynomialRegressionLine, poly)

def calibration(positionArray):
    #picks out the most efficient of the regression types (linear or polynomial)

    #setting up the two forms of regression
    linearRegressionLine = linearRegression(positionArray, percentArray)
    (polynomialRegressionLine, poly) = polynomialRegression(positionArray, percentArray, 2)

    #doing predictions with the input data
    linearPredict = linearRegressionLine.predict(positionArray)
    polynomialPredict = polynomialRegressionLine.predict(poly.fit_transform(positionArray))

    #picking the regression line that is the most accurate (with the lowest R score)
    if r2_score(percentArray, linearPredict) <= r2_score(percentArray, polynomialPredict):
        return ("linear", linearRegressionLine)
    else:
        return ("polynomial", polynomialRegressionLine, poly)


