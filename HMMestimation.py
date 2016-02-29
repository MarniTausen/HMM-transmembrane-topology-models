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
    def __init__(self, hmmdict):
        self.d = hmmdict
        self.states = {self.d['hidden'][i]:i for i in range(len(self.d['hidden']))}
        self.obs = {self.d['observables'][i]:i for i in range(len(self.d['observables']))}
        self.pi = self.d['pi']
        self.trans = matrix(self.makenested(self.d['transitions'], 3))
        self.emi = matrix(self.makenested(self.d['emissions'], 3))

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

files = ["Dataset160/set160.0.labels.txt", "Dataset160/set160.1.labels.txt", "Dataset160/set160.2.labels.txt", "Dataset160/set160.3.labels.txt",
         "Dataset160/set160.4.labels.txt", "Dataset160/set160.5.labels.txt", "Dataset160/set160.6.labels.txt", "Dataset160/set160.7.labels.txt",
         "Dataset160/set160.8.labels.txt"]

trainingdata = {}

for i in files:
    for k, v in loadseq(i).items():
        trainingdata[k] = v

#print trainingdata

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
print emission 

tot = 0
for l in emission:
    for j in l:
        tot = tot + j
emission = map(lambda x: map(lambda y: y/float(tot),x) , emission)

# transitions
row = {'i': 0, 'M': 1, 'o': 2, 'x': 3}
t = 0
transition = [[0 for j in range(4)] for i in range(4)]
for values in trainingdata.values():
    for h in range(len(values[1])-1):
       transition[row[values[1][h]]][row[values[1][h+1]]] += 1
       t += 1

transition = map(lambda x: map(lambda y: y/float(t),x) , transition)

###########
# Training by counting for 4 states module 
t = 0
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
        t += 1
print trans4

# we need to take into account the M followed by M, when is in each 4 statement
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

