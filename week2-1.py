import sys
import os
import random
from random import randint

def coinflip():
    return randint(0,1)

def experiment(numcoins = 1000, numflip = 10):
    coins = []
    
    for x in range(0, numcoins):
        results = []
        for y in range(0,numflip):
            results.append(coinflip())
            
        coins.append(results)
        
       
    v1 = sum(coins[0])/10.0
    #print "coin[0]: %s ; v1: %s" % (coins[0], v1)
    randnum = randint(0,numcoins -1)
    vrand = sum(coins[randnum])/10.0
    #print "coin[rand]: %s ; vrand: %s" % (coins[randnum], vrand)
    lowestprob = 1
    lowestprobnum = -1
    for x in range(0, numcoins -1):
        vcur = sum(coins[x])/10.0
        if vcur < lowestprob:
            lowestprob = vcur
            lowestprobnum = x
    vmin = sum(coins[lowestprobnum])/10.0
    #print "coin[lowestprobnum]: %s ; vmin: %s" % (coins[lowestprobnum], vmin)        
    return v1, vrand, vmin
numex = 100000.0
vmins = []
v1s = []
vrands = []

for x in range(0, numex):
    v1, vrand, vmin = experiment()
    v1s.append(v1)
    vrands.append(vrand)
    vmins.append(vmin)

avgv1 = sum(v1s)/numex
avgvrand = sum(vrands)/numex
avgvmin = sum(vmins)/numex

print avgv1, avgvrand, avgvmin
#results for 100,000 test runs: 0.50062, 0.500857, 0.037804