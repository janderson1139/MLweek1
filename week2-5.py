#! /usr/bin/python
#
# This is support material for the course "Learning from Data" on edX.org
# https://www.edx.org/course/caltechx/cs1156x/learning-data/1120
#
# The software is intented for course usage, no guarantee whatsoever
# Date: Sep 30, 2013
#
# Template for a LIONsolver parametric table script.
#
# Generates a table based on input parameters taken from another table or from user input
#
# Syntax:
# When called without command line arguments:
#    number_of_inputs
#    name_of_input_1 default_value_of_input_1
#    ...
#    name_of_input_n default_value_of_input_n
# Otherwise, the program is invoked with the following syntax:
#    script_name.py input_1 ... input_n table_row_number output_file.csv
# where table_row_number is the row from which the input values are taken (assume it to be 0 if not needed)
#
# To customize, modify the output message with no arguments given and insert task-specific code
# to insert lines (using tmp_csv.writerow) in the output table.

import sys
import os
import random
import numpy as np
from numpy import arange, array, ones, linalg

#############################################################################
#
# Task-specific code goes here.
#

# The following function is a stub for the perceptron training function required in Exercise1-7 and following.
# It currently generates random results.
# You should replace it with your implementation of the
# perceptron algorithm (we cannot do it otherwise we solve the homework for you :)
# This functon takes the coordinates of the two points and the number of training samples to be considered.
# It returns the number of iterations needed to converge and the disagreement with the original function.


def YatX(x, m, b=0):
    return m * x + b


def signF(xtotest, m, b):
    val = m * xtotest + b
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


def targetFunction(x1,y1,x2,y2,x3,y3):
    u = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    if u >= 0:
        return 1
    elif u < 0:
        return -1


def plaG(w, x, y):
    val = w[0] * 1 + w[1] * x + w[2] * y
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


def gen_points(num_data_points):
    points = np.random.uniform(-1, 1, (num_data_points, 2))
    x0 = np.ones((num_data_points, 1)) #add the artificial coordinate x0 = 1
    points = np.append(x0, points, axis=1)
    return points


def perceptron_training(x1, y1, x2, y2, training_size):
    m = (y2 - y1) / (x2 - x1)
    #y=mx +b; b = y-mx
    b = y1 - m * x1
    datapoints = gen_points(training_size)
    ydatapoints = []
    for datapoint in datapoints:
        yn = targetFunction(x1, y1, x2, y2, datapoint[1], datapoint[2])
        ydatapoints.append(yn)
    ydatapoints = np.asarray(ydatapoints)

    w = linalg.lstsq(datapoints, ydatapoints)[0]

    #Lets find Ein
    numwrong = 0
    for datapoint in datapoints:
        yn = targetFunction(x1, y1, x2, y2, datapoint[1], datapoint[2])
        gyn = plaG(w, datapoint[1], datapoint[2])
        if yn != gyn:
            numwrong += 1
    ein = float(numwrong) / training_size

    #lets find Eout
    numverify = 1000
    verifypoints = []
    numwrong = 0
    for num in range(1, numverify + 1):
        verifypoints.append((random.uniform(-1, 1), random.uniform(-1, 1)))
        datapoint = verifypoints[num - 1]
        yn = targetFunction(x1, y1, x2, y2, datapoint[0], datapoint[1])
        gyn = plaG(w, datapoint[0], datapoint[1])

        if yn != gyn:
            numwrong += 1
    eout = float(numwrong) / numverify


    allgood = 0
    numberIterations = 0
    while allgood != 1:
        random.shuffle(datapoints)

        allgood = 1
        for datapoint in datapoints:
            yn = targetFunction(x1, y1, x2, y2, datapoint[1], datapoint[2])
            gyn = plaG(w, datapoint[1], datapoint[2])
            #print ("iteration %s: yn = %s , Gyn = %s") % (numberIterations, yn, gyn)

            if (yn != gyn):
                numberIterations += 1
                allgood = 0
                w[0] += yn * 1
                w[1] += yn * datapoint[1]
                w[2] += yn * datapoint[2]
                break
    #Ein Section

    #return (int (random.gauss(100, 10)), random.random() / training_size)
    return ein, eout, numberIterations


tests = 1000
points = 10

# Repeat the experiment n times (tests parameter) and store the result of each experiment in one line of the output table
totalEin = 0
totalEout = 0
totalNumIterations = 0
for t in range(1, tests + 1):
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    Ein, Eout, NumIterations = perceptron_training(x1, y1, x2, y2, points)
    totalEin += Ein
    totalEout += Eout
    totalNumIterations += NumIterations

print "Average Ein: %s Average Eout: %s Average Iterations: %s" % ((totalEin/tests), (totalEout/tests), (totalNumIterations*1.0/tests))
