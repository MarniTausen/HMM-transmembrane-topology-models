from numpy import matrix
import numpy as np

from HMMdecoding import *

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


#priori values 
priori = []
for values in trainingdata.values():
    priori.append(values[1][0])
    #print len(set(values[0]))

priori = ''.join(priori)
i = priori.count('o')/float(len(priori))
o = priori.count('i')/float(len(priori))
print i+o
pi = [i, 0, o] # i M o

row = {'i': 0, 'M': 1, 'o': 2}
col = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4, 'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9, 'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14, 'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19}
# emissions: I M O
emission = [[0 for j in range(20)] for i in range(3)]
print emission 
for values in trainingdata.values():
    for o, s in zip(values[0], values[1]): #hidden states
        emission[row[s]][col[o]] += 1

# transitions
row = {'i': 0, 'M': 1, 'o': 2, 'x': 3}
transition = [[0 for j in range(3)] for i in range(3)]
for values in trainingdata.values():
    for h in range(len(values[1])-1):
       transition[row[values[1][h]]][row[values[1][h+1]]] += 1

print '\nModel 1: \n'
print pi
print transition
print emission
###########
# Training by counting for 4 states module: changing the characters
iM = False

themap = {'i': '0', 'o': '2'}

print trainingdata['5H2A_CRIGR'][1]

trans4 = [[0 for j in range(4)] for i in range(4)]
for keys, values in trainingdata.items():
    mapping = []
    for h in range(len(values[1])-1):
        # from i to M
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
        elif values[1][h] == 'M' and row[values[1][h+1]] != 'M':
            if iM==True:
                mapping.append('1')
            if iM==False:
                mapping.append('3')
        # i to i and o to o
        else:
            mapping.append(themap[values[1][h]])
    trainingdata[keys] = (values[0], "".join(mapping))

print trainingdata['5H2A_CRIGR'][1]

for values in trainingdata.values():
    for h in range(len(values[1])-1):
       trans4[int(values[1][h])][int(values[1][h+1])] += 1

#print trainingdata['CVAA_ECOLI']
#print '\nTransmission for 4 state model'
#for i in trans4:
  #  print i

# emissions for 4 statements:
# emissions: I M O
emis4 = [[0 for j in range(20)] for i in range(4)]
for values in trainingdata.values():
    em = []
    for j in zip(values[0], values[1]): 
        em.append(j) 
    for i in range(len(em)-1):
        emis4[int(em[i][1])][col[em[i][0]]] += 1 

print '\nModel 2 - 4 states: \n'
print pi
print trans4
print emis4

d2 = {}
d2['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
d2['hidden'] = ['i', 'M', 'o']
d2['pi'] = pi
d2['transitions'] = transition
d2['emissions'] = emission
hmm = HMMObject(d2, True)
print hmm
# Normalizing trans by col
for i in range(0,3):
    summ = np.sum(hmm.trans[i,:])   
    hmm.trans[i,:] = hmm.trans[i,:]/float(summ)
#print hmm.trans

# Normalizing emis by row
for i in range(0,3):
    summ = np.sum(hmm.emi[i,:])    
    hmm.emi[i,:] = hmm.emi[i,:]/float(summ)
#print hmm.emi

print hmm

velog = np.vectorize(elog)

hmm.emi = velog(hmm.emi)
hmm.trans = velog(hmm.trans)
hmm.pi = velog(hmm.pi)

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

d4 = {}
d4['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
d4['hidden'] = ['0', '1', '2', '3']
d4['pi'] = pi+[0]
d4['transitions'] = trans4
d4['emissions'] = emis4
hmm4 = HMMObject(d4, True, {'0': 'i', '1': 'M', '2': 'o', '3': 'M'})

print hmm4

for i in range(0,4):
    summ = np.sum(hmm4.trans[i,:])   
    hmm4.trans[i,:] = hmm4.trans[i,:]/float(summ)
#print hmm.trans

# Normalizing emis by row
for i in range(0,4):
    summ = np.sum(hmm4.emi[i,:])    
    hmm4.emi[i,:] = hmm4.emi[i,:]/float(summ)

print hmm4
    
hmm4.emi = velog(hmm4.emi)
hmm4.trans = velog(hmm4.trans)
hmm4.pi = velog(hmm4.pi)

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
