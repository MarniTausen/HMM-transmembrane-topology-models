from numpy import matrix
import numpy as np
from cross_validation import *
from HMMdecoding import *
from Manystatesmodel import *

# Printing out the hidden states + observations + loglikehood
print 'Viterbi decoding output' 

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
        if summ==0: continue
        hmm.trans[i,:] = hmm.trans[i,:]/float(summ)

    for i in range(0,hmm.emi.shape[0]):
        summ = np.sum(hmm.emi[i,:])
        if summ==0: continue
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

def decoding_save(algorithm, sets, hmm, outfile, convert=False):
    output = ""
    for k in sets:
        temp_viterbi = algorithm(sets[k][0], hmm)
        seq = temp_viterbi
        if convert==True: seq = convertback(temp_viterbi, hmm)
        output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, sets[k][0], seq, loglikelihood((sets[k][0], temp_viterbi), hmm))
    file = open(outfile, "w")
    file.write(output)
    file.close()

decoding_save(Viterbi, set9, hmm, 'output_set9.txt')
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

for k in set9:
    temp_viterbi = Viterbi(set9[k][0], hmm4)
    output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, set9[k][0], convertback(temp_viterbi, hmm4),
                                                         loglikelihood((set9[k][0], temp_viterbi), hmm4))
file = open('output_set9_hmm4.txt', "w")
file.write(output)
file.close()

os.system("python compare_tm_pred.py Dataset160/set160.9.labels.txt output_set9_hmm4.txt")

# Cross Validation for 3 states

files = ["Dataset160/set160.0.labels.txt", "Dataset160/set160.1.labels.txt", "Dataset160/set160.2.labels.txt", "Dataset160/set160.3.labels.txt",
         "Dataset160/set160.4.labels.txt", "Dataset160/set160.5.labels.txt", "Dataset160/set160.6.labels.txt", "Dataset160/set160.7.labels.txt",
         "Dataset160/set160.8.labels.txt", "Dataset160/set160.9.labels.txt"]
testing =[]
train = []
cv = []
testingdata = {}
state3 = {'i': 0, 'M': 1, 'o': 2}
obsdict = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4, 'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9, 'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14, 'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19}
def CV3state(files, decoding=Viterbi):
    cv = []
    for j in range(0,10):
        train = files[:j] + files[j+1:]
        testing = files[j]
        trainingdata = {}
        for i in train:
            for k, v in loadseq(i).items():
                trainingdata[k] = v
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
        decoding_save(decoding, test, hmm, 'output_testing.txt')
        cv.append(cross_validation(testing, 'output_testing.txt'))
    return cv

def mean(cv):
    summs = []
    means = 0
    for i in range(len(cv)):
        if cv[i][3]!=float('inf'):
            summs += [cv[i][3]]
    means = sum(summs)/len(summs)
    return means

def var(cv, mean):
    var = 0
    for i in range(len(cv)):
        var += (cv[i][3] - mean)**2
    var = var/float(len(cv))
    return var

cv = CV3state(files)

print 'The mean of the cross validation (3 states) AC results is:'
mean3 = mean(cv)
print mean3
print 'The variance of the cross validation (3 states) AC results is:'
print var(cv, mean3)
#print testingdata

# Cross validation for 4 states
def CV4state(files, decoding=Viterbi):
    cv4 = []
    for j in range(0,10):
        train = files[:j] + files[j+1:]
        testing = files[j]
        trainingdata = {}
        for i in train:
            for k, v in loadseq(i).items():
                trainingdata[k] = v
    
        trainingdata = map4state(trainingdata)

        state4 = {'0': 0, '1': 1, '2': 2, '3': 3}
        d4 = {}
        d4['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
        d4['hidden'] = ['0', '1', '2', '3']
        d4['pi'] = countPriori(trainingdata, ['0', '1', '2', '3'])
        d4['transitions'] = countTransitions(trainingdata, state4)
        d4['emissions'] = countEmissions(trainingdata, state4, obsdict)
        hmm4 = HMMObject(d4, True, {'0': 'i', '1': 'M', '2': 'o', '3': 'M'})

        hmm4 = normalize(hmm4)
        hmm4 = logtransform(hmm4)
        
        test = loadseq(testing)
        decoding_save(decoding, test, hmm4, 'output_testing.txt', True)
        cv4.append(cross_validation(testing, 'output_testing.txt'))
    return cv4

cv4 = CV4state(files)

print
print 'The mean of the cross validation (4 states) AC results is:'
mean4 = mean(cv4)
print mean4
print 'The variance of the cross validation (4 states) AC results is:'
print var(cv4, mean4)

print
print "Results using Posterior decoding instead of Viterbi:"
print

cv3 = CV3state(files, Posterior)

print
print 'The mean of the cross validation (3 states) AC results is:'
mean3 = mean(cv3)
print mean3
print 'The variance of the cross validation (3 states) AC results is:'
print var(cv3, mean3)

cv4 = CV4state(files, Posterior)

print
print 'The mean of the cross validation (4 states) AC results is:'
mean4 = mean(cv4)
print mean4
print 'The variance of the cross validation (4 states) AC results is:'
print var(cv4, mean4)

## 73-state model.

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

decoding_save(Viterbi, set9, hmmTMH, 'output_set9_hmmTMH.txt', True)

os.system("python compare_tm_pred.py Dataset160/set160.9.labels.txt output_set9_hmmTMH.txt")

def CVTMHstate(files, decoding=Viterbi):
    cv = []
    for j in range(0,10):
        train = files[:j] + files[j+1:]
        testing = files[j]
        trainingdata = {}
        for i in train:
            for k, v in loadseq(i).items():
                trainingdata[k] = v
    
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

        test = loadseq(testing)
        decoding_save(decoding, test, hmmTMH, 'output_testing.txt', True)
        cv.append(cross_validation(testing, 'output_testing.txt'))
    return cv


print
print 'Results of the 73-state model:'
print

print 'Viterbi'

cv = CVTMHstate(files)

print
print 'The mean of the cross validation (4 states) AC results is:'
themean = mean(cv)
print themean
print 'The variance of the cross validation (4 states) AC results is:'
print var(cv, themean)

print
print 'Posterior'

cv = CVTMHstate(files, Posterior)

print
print 'The mean of the cross validation (4 states) AC results is:'
themean = mean(cv)
print themean
print 'The variance of the cross validation (4 states) AC results is:'
print var(cv, themean)


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
