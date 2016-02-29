from numpy import matrix
import numpy as np

# Object for storing all of the HMM (Hidden Markov Model) data inside.
# Takes the output from loadHMM() and makes the data more accessible
class HMMObject(object):

    """
    Contains the following variables: states (latent states), obs (observables),
    pi (priori probabilities), trans (transition probabilities), emi (emission probabilities)"""

    # Initialization of object, preparing all of the datatypes.
    # States and obs get converted into a dictionary with the letters as keys,
    # and the values are the corresponding indices.
    # Trans and emi get converted into a nested list, with 3 internal lists.
    def __init__(self, hmmdict, nested=False):
        self.d = hmmdict
        self.states = {self.d['hidden'][i]:i for i in range(len(self.d['hidden']))}
        self.obs = {self.d['observables'][i]:i for i in range(len(self.d['observables']))}
        self.pi = self.d['pi']
        if nested==False:
            self.trans = matrix(self.makenested(self.d['transitions'], 3))
            self.emi = matrix(self.makenested(self.d['emissions'], 3))
        else:
            self.trans = matrix(self.d['transitions'], dtype="float")
            self.emi = matrix(self.d['emissions'], dtype="float")

    def __str__(self):
        output = "Hidden Markov Model \n\n"
        output += "Hidden states: \n"
        output += " ".join([i for i in self.states.keys()])+"\n\n"
        output += "Observables: \n"
        output += " ".join([i for i in self.obs.keys()])+"\n\n"
        output += "Pi: \n"
        output += " ".join(["%.5f" % i for i in self.pi])+"\n\n"
        output += "Transitions: \n"
        for i in range(self.trans.shape[1]):
            output += " ".join(["%.5f" % float(j) for j in self.trans[:,i]])+"\n"
        output += "\nEmissions: \n"
        for i in range(self.emi.shape[0]):
            output += " ".join(["%.5f" % float(j) for j in np.nditer(self.emi[i,:])])+"\n"
        return output
        
    # Function splits a list into a nested list, into r parts. (r meaning rows)
    def makenested(self, x, r):
        n = len(x)/r
        result = []
        for i in range(r):
            result.append(x[i*n:n*(i+1)])
        return result

# Wrapper for the log function
def elog(x):
    from math import log
    if x == 0:
        return float("-inf")
    return log(x)

# Loading the hidden markov model (hmm) data.
# First by splitting the for each of the names.
# Then collecting all of the data with the right labels in a dictionary.
# Then the necessary data conversion
def loadHMM(filename):
    import re
    # Loading the data
    rawdata = open(filename, "r").read().replace("\n", " ")
    # Splitting by name
    splitdata = re.split("hidden|observables|pi|transitions|emissions", rawdata)
    splitdata = [i.strip().split(" ") for i in splitdata[1:]]
    # Collecting the data in a dictionary
    labels = ["hidden", "observables", "pi", "transitions", "emissions"]
    d = {labels[i]:splitdata[i] for i in range(len(splitdata))}
    # Data conversion, from string to float and from probability to log.
    d['pi'] = [elog(float(i)) for i in d['pi']]
    d['transitions'] = [elog(float(i)) for i in d['transitions']]
    d['emissions'] = [elog(float(i)) for i in d['emissions']]
    # Inputing the data into the HMMObject class.
    return HMMObject(d)

# Loading the sequence data in a fasta format, with sequence and latent states separated by #.
def loadseq(filename):
    rawdata = open(filename, "r").read().split(">")
    splitdata = [i.split("#") for i in rawdata[1:]]
    splitdata = [i[0].split("\n",1)+[i[1]] for i in splitdata]
    return {i[0]:(i[1].strip(), i[2].strip()) for i in splitdata}

# Loading the sequence data in a fasta format, with sequence and headers.
def readFasta(filename):
    import re
    raw = open(filename, "r").read().replace("\n", " ").strip().split(">")
    split = [re.split("\s*", i) for i in raw[1:]]
    return {i[0]:i[1] for i in split}

# Calculating the log likelihood of the joint probability
# The parameters of the HMM model must be log transformed.
def loglikelihood(seqpair, HMM):

    from math import log

    # Calculating for the initial pi and the initial emission
    result = HMM.emi[HMM.states[seqpair[1][0]],HMM.obs[seqpair[0][0]]]
    result += HMM.pi[HMM.states[seqpair[1][0]]]

    # Storing the previous state.
    prevstate = seqpair[1][0]

    # Iterating over all of the remaining observations and latent states
    for i in zip(seqpair[0], seqpair[1])[1:]:
        # Transitions
        result += HMM.trans[HMM.states[prevstate],HMM.states[i[1]]]
        # Emissions
        result += HMM.emi[HMM.states[i[1]],HMM.obs[i[0]]]
        prevstate = i[1]

    return result

def Viterbi(seq, hmm):
    # Initialize the omega table
    N = len(seq)
    M = np.zeros((len(hmm.states), N))
    M.fill(float("-inf"))

    # Fill in the first column.
    for k in hmm.states.values():
        M[k,0] = hmm.pi[k]+hmm.emi[k,hmm.obs[seq[0]]]

    # Fill in the remaining columns in the table.
    n = 1
    for i in seq[1:]:
        o = hmm.obs[i]
        for k in hmm.states.values():
            if hmm.emi[k,o]!=float("-inf"):
                for j in hmm.states.values():
                    if hmm.trans[j,k]!=float("-inf"):
                        M[k,n] = max([M[k,n], M[j, n-1]+hmm.emi[k,o]+hmm.trans[j,k]])
        n += 1

    # Backtracking:
    z = ['' for i in range(len(seq))]

    # Find the last max:
    z[N-1] = hmm.states.keys()[M[:,N-1].argmax()]
    for n in range(N-1)[::-1]:
        z[n] = hmm.states.keys()[M[:,n].argmax()]

    z = ["o" if i=="i" else i for i in z]

    #Backtrack.
    for n in range(N-1)[::-1]:
        temp = np.array([float("-inf") for i in range(len(hmm.states))])
        o, ns = hmm.obs[seq[n+1]], hmm.states[z[n+1]]
        for i in hmm.states.values():
            temp[i] = hmm.emi[i,o]+M[i,n]+hmm.trans[i, ns]
        z[n] = hmm.states.keys()[temp.argmax()]
        #print

    for i in range(len(z)):
        if z[i]=="oM" or z[i]=="iM":
            z[i] = "M"
        
    return "".join(z)

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
# Training by counting for 4 states module 
iM = False
trans4 = [[0 for j in range(4)] for i in range(4)]
for values in trainingdata.values():
    for h in range(len(values[1])-1):
        # from i to M
        if row[values[1][h]] == 0 and row[values[1][h+1]] == 1:
            trans4[0][1] += 1
            iM = True
        # from o to M
        elif row[values[1][h]] == 2 and row[values[1][h+1]] == 1:
            trans4[2][3] += 1
            iM = False
        # M to M
        elif row[values[1][h]] == 1 and row[values[1][h+1]] == 1:
            if iM==True:
                trans4[1][1] += 1
            if iM==False:
                trans4[3][3] += 1
        # M to i or o
        elif row[values[1][h]] == 1 and row[values[1][h+1]] != 1:
            if iM==True:
                trans4[1][0] += 1
            if iM==False:
                trans4[3][2] += 1
        # i to i and o to o
        else:
            trans4[row[values[1][h]]][row[values[1][h+1]]] += 1

# emissions for 4 statements:
# emissions: I M O
em = []
emis4 = [[0 for j in range(20)] for i in range(4)]
for values in trainingdata.values():
    for j in zip(values[0], values[1]): 
        em.append(j) 
for i in range(len(em)-1):
    if em[i][1] == 'M' and em[i-1][1] == 'i':
        emis4[1][col[em[i][0]]] += 1 
        iM = True
    if em[i][1] == 'M' and em[i-1][1] == 'o':
        emis4[3][col[em[i][0]]] += 1 
        iM = False
    if em[i][1] == 'M' and em[i-1][1] == 'M':
        if iM==True:
            emis4[1][col[em[i][0]]] += 1
        if iM==False:
            emis4[3][col[em[i][0]]] += 1
    if em[i][1] == 'M' and em[i-1][1] != 'M':
        if iM==True:
            emis4[1][col[em[i][0]]] += 1
        if iM==False:
            emis4[3][col[em[i][0]]] += 1
    else:
        emis4[row[em[i][1]]][col[em[i][0]]] += 1

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
    summ = np.sum(hmm.trans[:,i])   
    hmm.trans[:,i] = hmm.trans[:,i]/float(summ)
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

d3 = {}
d3['observables'] = ['A', 'C', 'E', 'D', 'G', 'F', 'I','H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'] 
d3['hidden'] = ['i', 'iM', 'o', 'oM']
d3['pi'] = pi+[0]
d3['transitions'] = trans4
d3['emissions'] = emis4
hmm4 = HMMObject(d3, True)

for i in range(0,4):
    summ = np.sum(hmm4.trans[:,i])   
    hmm4.trans[:,i] = hmm4.trans[:,i]/float(summ)
#print hmm.trans

# Normalizing emis by row
for i in range(0,4):
    summ = np.sum(hmm4.emi[i,:])    
    hmm4.emi[i,:] = hmm4.emi[i,:]/float(summ)

hmm4.emi = velog(hmm4.emi)
hmm4.trans = velog(hmm4.trans)
hmm4.pi = velog(hmm4.pi)

print hmm4

print hmm4.obs

output = ""

for k in set9:
    temp_viterbi = Viterbi(set9[k][0], hmm4)
    output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (k, set9[k][0], temp_viterbi, 0)
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
