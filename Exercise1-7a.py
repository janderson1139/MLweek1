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

#
# If there are not enough parameters, optionally write out the number of required parameters
# followed by the list of their names and default values. One parameter per line,
# name followed by tab followed by default value.
# LIONsolver will use this list to provide a user friendly interface for the component's evaluation.
#
if len(sys.argv) < 2:
    sys.stdout.write("2\nNumber of tests\t1000\nNumber of training points\t10\n")
    sys.exit(0)
#
# Retrieve the input parameters, the input row number, and the output filename.
#
in_parameters = [float(x) for x in sys.argv[1:-2]]
in_rownumber = int(sys.argv[-2])
out_filename = sys.argv[-1]

#
# Retrieve the output filename from the command line; create a temporary filename
# and open it, passing it through the CSV writer
#
tmp_filename = out_filename + "_"
tmp_file = open(tmp_filename, "w")

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
    return (m * x + b)


def signF(xtotest, m, b):
    val = m * xtotest + b
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


def plaG(w, x, y):
    val = w[0] * 1 + w[1] * x + w[2] * y
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


def perceptron_training(x1, y1, x2, y2, training_size):
    m = (y2 - y1) / (x2 - x1)
    #y=mx +b; b = y-mx
    b = y1 - m * x1

    datapoints = []
    w = [0, 0, 0]
    for num in range(1, training_size + 1):
        datapoints.append((random.uniform(-1, 1), random.uniform(-1, 1)))

    allgood = 0
    numberIterations = 0
    while allgood != 1:

        allgood = 1
        for datapoint in datapoints:
            yn = signF(datapoint[0], m, b)
            gyn = plaG(w, datapoint[0], datapoint[1])
            #print ("iteration %s: yn = %s , Gyn = %s") % (numberIterations, yn, gyn)

            if (yn != gyn):
                numberIterations += 1
                allgood = 0
                w[0] += yn * 1
                w[1] += yn * datapoint[0]
                w[2] += yn * datapoint[1]
                break
        #lets find disagreement
    numverify = 10000
    verifypoints = []
    numwrong = 0
    for num in range(1, numverify + 1):
        verifypoints.append((random.uniform(-1, 1), random.uniform(-1, 1)))
        datapoint = verifypoints[num - 1]
        yn = signF(datapoint[0], m, b)
        gyn = plaG(w, datapoint[0], datapoint[1])

        if (yn != gyn):
            numwrong += 1
    prob = float(numwrong) / numverify
    print ("iters: %s numwrong: %s prob: %s") % (numberIterations, numwrong, prob)

    #return (int (random.gauss(100, 10)), random.random() / training_size)
    return (numberIterations, prob)


tests = int(in_parameters[0])
points = int(in_parameters[1])

# Write the header line in the output file, in this case the output is a 3-columns table containing the results
# of the experiments
# The syntax  name::type  is used to identify the columns and specify the type of data
header = "Test number::label,Number of iterations::number,Disagreement::number\n"
tmp_file.write(header)


# Repeat the experiment n times (tests parameter) and store the result of each experiment in one line of the output table
totaliterations = 0
totaldisagreement = 0
for t in range(1, tests + 1):
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    iterations, disagreement = perceptron_training(x1, y1, x2, y2, points)
    line = str(t) + ',' + str(iterations) + ',' + str(disagreement) + '\n'
    totaliterations += iterations
    totaldisagreement += disagreement
    tmp_file.write(line)
print "Average iterations: %s   Average Disagreement %s" % ((totaliterations / tests), (totaldisagreement / tests))
#
#############################################################################

#
# Close all input files and the temporary output file.
#
tmp_file.close()

#
# Rename the temporary output file into the final one.
# It's important that the output file only appears when it is complete,
# otherwise LIONsolver might read an incomplete table.
#
os.rename(tmp_filename, out_filename)
