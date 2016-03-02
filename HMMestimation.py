from numpy import matrix
import numpy as np
from cross_validation import *
from HMMdecoding import *
from Manystatesmodel import *

# Printing out the hidden states + observations + loglikehood
print 'Viterbi decoding output' 
#for key in sorted(sequences):
 #   temp_viterbi = Viterbi(sequences[key], hmm)
  #  print'>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (key, sequences[key], temp_viterbi, loglikelihood((sequences[key], temp_viterbi), hmm))


files = ["Dataset160/set160.0.labels.txt", "Dataset160/set160.1.labels.txt", "Dataset160/set160.2.labels.txt", "Dataset160/set160.3.labels.txt",
         "Dataset160/set160.4.labels.txt", "Dataset160/set160.5.labels.txt", "Dataset160/set160.6.labels.txt", "Dataset160/set160.7.labels.txt",
         "Dataset160/set160.8.labels.txt"]

trainingdata = {}
sequences = {}
for i in files:
    for k, v in loadseq(i).items():
        trainingdata[k] = v
        sequences[k] = v[0] # To be used in the Viterbi algorithm 

def countPriori(data, states):
    priori = []
    for values in data.values():
        priori.append(values[1][0])
    priori = ''.join(priori)
    counts = [priori.count(i)/float(len(priori)) for i in states]
    return counts

def countEmissions(data, states, obs):
    emission = [[0 for j in range(len(obs))] for i in range(len(states))]
    for values in data.values():
        for o, s in zip(values[0], values[1]): #hidden states
            emission[states[s]][obs[o]] += 1
    return emission

def countTransitions(data, states):
    transition = [[0 for j in range(len(states))] for i in range(len(states))]
    for values in data.values():
        for h in range(len(values[1])-1):
            transition[states[values[1][h]]][states[values[1][h+1]]] += 1
    return transition

###########
# Training by counting for 4 states module: changing the characters
def map4state(data):
    themap = {'i': '0', 'o': '2'}
    iM = False
    #states = {'i': 0, 'M': 1, 'o': 2}
    for keys, values in data.items():
        mapping = []
        for h in range(len(values[1])):
            # from i to M
            if h<(len(values[1])-1):
                if values[1][h] == 'i' and values[1][h+1] == 'M':
                    mapping.append('0')
                    iM = True
                    # from o to M
                elif values[1][h] == 'o' and values[1][h+1] == 'M':
                    mapping.append('2')
                    iM = False
                    # M to M
                elif values[1][h] == 'M' and values[1][h+1] == 'M':
                    if iM==True:
                        mapping.append('1')
                    if iM==False:
                        mapping.append('3')
                        # M to i or o
                elif values[1][h] == 'M' and values[1][h+1] != 'M':
                    if iM==True:
                        mapping.append('1')
                    if iM==False:
                        mapping.append('3')
                        # i to i and o to o
                else:
                    mapping.append(themap[values[1][h]])
            else:
                if values[1][h]=='M':
                    if iM==True:
                        mapping.append('1')
                    if iM==False:
                        mapping.append('3')
                else:
                    mapping.append(themap[values[1][h]])
        data[keys] = (values[0], "".join(mapping))
    return data

state3 = {'i': 0, 'M': 1, 'o': 2}
obsdict = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4, 'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9, 'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14, 'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19}

d2 = {}
d2['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
d2['hidden'] = ['i', 'M', 'o']
d2['pi'] = countPriori(trainingdata, ['i', 'M', 'o'])
d2['transitions'] = countTransitions(trainingdata, state3)
d2['emissions'] = countEmissions(trainingdata, state3, obsdict)
hmm = HMMObject(d2, True)
print hmm

def normalize(hmm):
    for i in range(0,hmm.trans.shape[0]):
        summ = np.sum(hmm.trans[i,:])
        if summ==0:
            continue
        hmm.trans[i,:] = hmm.trans[i,:]/float(summ)

    for i in range(0,hmm.emi.shape[0]):
        summ = np.sum(hmm.emi[i,:])
        if summ==0:
            continue
        hmm.emi[i,:] = hmm.emi[i,:]/float(summ)

    return hmm

hmm = normalize(hmm)

print hmm

def logtransform(hmm):
    velog = np.vectorize(elog)

    hmm.emi = velog(hmm.emi)
    hmm.trans = velog(hmm.trans)
    hmm.pi = velog(hmm.pi)

    return hmm

hmm = logtransform(hmm)

print hmm

output = str()
for k in sequences:
    temp_viterbi = Viterbi(sequences[k], hmm)
    output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, sequences[k], temp_viterbi, loglikelihood((sequences[k], temp_viterbi), hmm))
file = open('output_viterbi.txt', "w")
file.write(output)
file.close()

set9 = loadseq("Dataset160/set160.9.labels.txt")

output = ""

for k in set9:
    temp_viterbi = Viterbi(set9[k][0], hmm)
    output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, set9[k][0], temp_viterbi, loglikelihood((set9[k][0], temp_viterbi), hmm))
file = open('output_set9.txt', "w")
file.write(output)
file.close()

import os

os.system("python compare_tm_pred.py Dataset160/set160.9.labels.txt output_set9.txt")

state4 = {'0': 0, '1': 1, '2': 2, '3': 3}
trainingdata = map4state(trainingdata)

d4 = {}
d4['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
d4['hidden'] = ['0', '1', '2', '3']
d4['pi'] = countPriori(trainingdata, ['0', '1', '2', '3'])
d4['transitions'] = countTransitions(trainingdata, state4)
d4['emissions'] = countEmissions(trainingdata, state4, obsdict)
hmm4 = HMMObject(d4, True, {'0': 'i', '1': 'M', '2': 'o', '3': 'M'})

print hmm4

hmm4 = normalize(hmm4)

print hmm4
    
hmm4 = logtransform(hmm4)

print hmm4

output = ""

def convertback(z, hmm):
    r = ""
    for i in z:
        r += hmm.mapper[i]
    return r

set9 = map4state(set9)

for k in set9:
    temp_viterbi = Viterbi(set9[k][0], hmm4)
    output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, set9[k][0], convertback(temp_viterbi, hmm4),
                                                         loglikelihood((set9[k][0], temp_viterbi), hmm4))
file = open('output_set9_hmm4.txt', "w")
file.write(output)
file.close()

os.system("python compare_tm_pred.py Dataset160/set160.9.labels.txt output_set9_hmm4.txt")

# Cross Validation

files = ["Dataset160/set160.0.labels.txt", "Dataset160/set160.1.labels.txt", "Dataset160/set160.2.labels.txt", "Dataset160/set160.3.labels.txt",
         "Dataset160/set160.4.labels.txt", "Dataset160/set160.5.labels.txt", "Dataset160/set160.6.labels.txt", "Dataset160/set160.7.labels.txt",
         "Dataset160/set160.8.labels.txt", "Dataset160/set160.9.labels.txt"]
testing =[]
train = []
cv = []
testingdata = {}
state3 = {'i': 0, 'M': 1, 'o': 2}
obsdict = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4, 'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9, 'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14, 'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19}
for j in range(0,10):
    train = files[:j] + files[j+1:]
    testing = files[j]
    trainingdata = {}
    for i in train:
        for k, v in loadseq(i).items():
            trainingdata[k] = v
            sequences[k] = v[0]  ##########################!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    d2 = {}
    d2['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
    d2['hidden'] = ['i', 'M', 'o']
    d2['pi'] = countPriori(trainingdata, ['i', 'M', 'o'])
    d2['transitions'] = countTransitions(trainingdata, state3)
    d2['emissions'] = countEmissions(trainingdata, state3, obsdict)
    hmm = HMMObject(d2, True)
    hmm = normalize(hmm)
    hmm = logtransform(hmm)

    test = loadseq(testing)
    #print hmm
    output = ""
    for k in test:
        temp_viterbi = Viterbi(test[k][0], hmm)
        output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, test[k][0], temp_viterbi, loglikelihood((test[k][0], temp_viterbi), hmm))
    file = open('output_testing.txt', "w")
    file.write(output)
    file.close()

    cv.append(cross_validation(testing, 'output_testing.txt'))

print cv
summs = 0
for i in range(len(cv)):
    summs += cv[i][3]
means = summs/len(cv)

var = 0
for i in range(len(cv)):
    var += (cv[i][3] - means)**2
var = var/float(len(cv))
print 'Mean of AC'
print means
print 'Variance of AC'
print var
#print testingdata








## 73-state model.

files = ["Dataset160/set160.0.labels.txt", "Dataset160/set160.1.labels.txt", "Dataset160/set160.2.labels.txt", "Dataset160/set160.3.labels.txt",
         "Dataset160/set160.4.labels.txt", "Dataset160/set160.5.labels.txt", "Dataset160/set160.6.labels.txt", "Dataset160/set160.7.labels.txt",
         "Dataset160/set160.8.labels.txt"]

trainingdata = {}
sequences = {}
for i in files:
    for k, v in loadseq(i).items():
        trainingdata[k] = v
        sequences[k] = v[0]

trainingdata = TMHmapping(trainingdata)

TMHstates = {str(i):i for i in range(73)}
TMHmap = {'0': 'i', '36': 'o'}
for i in range(1, 36)+range(37, 73):
    TMHmap[str(i)] = 'M'

dTMH = {}
dTMH['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
dTMH['hidden'] = [str(i) for i in range(73)]
dTMH['pi'] = countPriori(trainingdata, [str(i) for i in range(73)])
dTMH['transitions'] = countTransitions(trainingdata, TMHstates)
dTMH['emissions'] = countEmissions(trainingdata, TMHstates, obsdict)
hmmTMH = HMMObject(dTMH, True, TMHmap)

hmmTMH = normalize(hmmTMH)
hmmTMH = logtransform(hmmTMH)

set9 = loadseq("Dataset160/set160.9.labels.txt")
set9 = TMHmapping(set9)

for k in set9:
    temp_viterbi = Viterbi(set9[k][0], hmmTMH)
    output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, set9[k][0], convertback(temp_viterbi, hmmTMH),
                                                         loglikelihood((set9[k][0], temp_viterbi), hmmTMH))
file = open('output_set9_hmmTMH.txt', "w")
file.write(output)
file.close()

os.system("python compare_tm_pred.py Dataset160/set160.9.labels.txt output_set9_hmmTMH.txt")


# for j in range(0,10):
#     train = files[:j] + files[j+1:]
#     testing = files[j]
#     trainingdata = {}
#     for i in train:
#         for k, v in loadseq(i).items():
#             trainingdata[k] = v
#             sequences[k] = v[0]  ##########################!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     trainingdata = TMHmapping(trainingdata)
#     TMHstates = {str(i):i for i in range(73)}
#     TMHmap = {'0': 'i', '36': 'o'}
#     for i in range(1, 36)+range(37, 73):
#         TMHmap[str(i)] = 'M'
#     dTMH = {}
#     dTMH['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
#     dTMH['hidden'] = [str(i) for i in range(73)]
#     dTMH['pi'] = countPriori(trainingdata, [str(i) for i in range(73)])
#     dTMH['transitions'] = countTransitions(trainingdata, TMHstates)
#     dTMH['emissions'] = countEmissions(trainingdata, TMHstates, obsdict)
#     hmm = HMMObject(dTMH, True, TMHmap)

#     hmm = normalize(hmm)
#     hmm = logtransform(hmm)
    
#     test = loadseq(testing)
#     test = TMHmapping(test)
#     #print hmm
#     output = ""
#     for k in test:
#         temp_viterbi = Viterbi(test[k][0], hmm)
#         output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, test[k][0], convertback(temp_viterbi, hmm), loglikelihood((test[k][0], temp_viterbi), hmm))
#     file = open('output_testing.txt', "w")
#     file.write(output)
#     file.close()

#     cv.append(cross_validation(testing, 'output_testing.txt'))

# summs = 0
# for i in range(len(cv)):
#     summs += cv[i][3]
# means = summs/len(cv)

# var = 0
# for i in range(len(cv)):
#     var += (cv[i][3] - means)**2
# var = var/float(len(cv))
# print 'Mean of AC'
# print means
# print 'Variance of AC'
# print var

# Comparing - validation approach for 1 set of the data
 # python compare_tm_pred.py set160.0.labels.txt output_viterbi.txt 

#### 1. Train the 3-state model (iMo) on parts 0-8 of the training data using training-by-counting.

## Count each occurance of observation and hidden state, through all of the training data specified.

## Convert it into frequencies.

### Show the obtained model parameters (transition, emission, and start probabilities)

#### 2. Train the 4-state model. Recall that for this model, the given annotations does not correspond immediately to sequences of hidden states, as there are two states that we interpret as being in transmembrane helix (annotation M).

#### 3. Make a 10-flod experiment using the 3-state model, training-by-counting and Viterbi decoding for prediction.

### Show the AC compute by the compare_tm_pred.py for each fold, and show the mean and variance of the ACs over all 10 folds.

#### 4. Redo step 3 with the 4-state model.

#### 5. Redo step 3 and 4 using Posterior decoding. How does the results obtained by posterior decoding compare to the results obtained by Viterbi decoding?

#### 6. Redo steps 3-5 for any other models that you find relevant, e.g. the ones we talked about in class. What is the best AC (i.e. best mean over a 10-fold experiment) that you can obtain? How does you best model look like?
