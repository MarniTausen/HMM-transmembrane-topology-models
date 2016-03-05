from numpy import matrix
import numpy as np
from collections import OrderedDict

# Object for storing all of the HMM (Hidden Markov Model) data inside.
# Takes the output from loadHMM() and makes the data more accessible
class HMMObject(object):
    
    """
    Contains the following variables: states (latent states), obs (observables),
    pi (priori probabilities), trans (transition probabilities), emi (emission probabilities)"""

    mapped = False
    
    # Initialization of object, preparing all of the datatypes.
    # States and obs get converted into a dictionary with the letters as keys,
    # and the values are the corresponding indices.
    # Trans and emi get converted into a nested list, with 3 internal lists.
    def __init__(self, hmmdict, nested=False, mapper={}):
        self.d = hmmdict
        self.states = OrderedDict([(self.d['hidden'][i],i) for i in range(len(self.d['hidden']))])
        self.obs = OrderedDict([(self.d['observables'][i],i) for i in range(len(self.d['observables']))])
        self.pi = self.d['pi']
        if nested==False:
            self.trans = matrix(self.makenested(self.d['transitions'], 3))
            self.emi = matrix(self.makenested(self.d['emissions'], 3))
        else:
            self.trans = matrix(self.d['transitions'], dtype="float")
            self.emi = matrix(self.d['emissions'], dtype="float")
        if mapper!={}:
            self.mapped = True
            self.mapper = mapper

    def __str__(self):
        output = "Hidden Markov Model \n\n"
        output += "Hidden states: \n"
        if self.mapped==True:
            output += " ".join([i+"("+self.mapper[i]+")" for i in self.states.keys()])+"\n\n"
        else:
            output += " ".join([i for i in self.states.keys()])+"\n\n"
        output += "Observables: \n"
        output += " ".join([i for i in self.obs.keys()])+"\n\n"
        output += "Pi: \n"
        output += " ".join(["%.5f" % i for i in self.pi])+"\n\n"
        output += "Transitions: \n"
        for i in range(self.trans.shape[1]):
            output += " ".join(["%.5f" % float(j) for j in np.nditer(self.trans[i,:])])+"\n"
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

def eexp(x):
    from math import exp
    if x == float("-inf"):
        return 0
    return exp(x)

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

def Viterbi(seq, hmm, prob=False):
    # Initialize the omega table
    N = len(seq)
    M = np.zeros((len(hmm.states), N))
    M.fill(float("-inf"))

    if prob==True:
        P = np.zeros((len(hmm.states), N))
    
    # Fill in the first column.
    for k in range(len(hmm.states)):
        M[k,0] = hmm.pi[k]+hmm.emi[k,hmm.obs[seq[0]]]
        if prob==True: P[k,0] = eexp(hmm.pi[k]+hmm.emi[k,hmm.obs[seq[0]]])
        
    # Fill in the remaining columns in the table.
    for n in range(1, N):
        o = hmm.obs[seq[n]]
        for k in range(len(hmm.states)):
            if hmm.emi[k,o]!=float("-inf"):
                for j in hmm.states.values():
                    if hmm.trans[j,k]!=float("-inf"):
                        M[k,n] = max([M[k,n], M[j, n-1]+hmm.emi[k,o]+hmm.trans[j,k]])
                        if prob==True: P[k,0] = max([P[k,n], eexp(hmm.emi[k,o]+hmm.trans[j,k]]))
    
    # Backtracking:
    z = ['' for i in range(len(seq))]

    # Find the last max:
    z[N-1] = hmm.states.keys()[M[:,N-1].argmax()]

    #Backtrack.
    for n in range(N-1)[::-1]:
        o, ns = hmm.obs[seq[n+1]], hmm.states[z[n+1]]
        for k in range(len(hmm.states)):
            if M[k,n]+hmm.emi[ns,o]+hmm.trans[k, ns] == M[ns, n+1]:
                z[n] = hmm.states.keys()[k]
                break

    if prob==True:
        return P
            
    if len(hmm.states)<10:
        return "".join(z)
    else:
        return z

##################### Starting Posterior decoding #####################

# Making the logsum to transform the data
def LOGSUM(x, y): #the input is already log transformed
    if x == float("-inf"):
        return y
    if y == float("-inf"):
        return x
    if x > y:
        return x + elog(1 + eexp(y - x))
    else:
        return y + elog(1 + eexp(x - y))

def Posterior(seq, hmm):
    # Initializing the tables
    N = len(seq)
    A = np.zeros((len(hmm.states), N))
    A.fill(float("-inf"))
    B = np.zeros((len(hmm.states), N))
    B.fill(float("-inf"))

    # Filling up the alpha table:
    for k in hmm.states.values():
        A[k,0] = hmm.pi[k]+hmm.emi[k,hmm.obs[seq[0]]]

    for n in range(1,N):
        o = hmm.obs[seq[n]]
        for k in hmm.states.values():
            logsum = float("-inf")
            if hmm.emi[k,o]!=float("-inf"):
                for j in hmm.states.values():
                    if hmm.trans[j,k]!=float("-inf"):
                        logsum = LOGSUM(logsum, A[j, n-1]+hmm.trans[j,k])
            if logsum!=float("-inf"):
                logsum += hmm.emi[k,o]
            A[k,n] = logsum

    # Filling up the beta table:
    B[:,N-1] = elog(1)
    for n in range(0,N-1)[::-1]:
        o = hmm.obs[seq[n+1]]
        for k in hmm.states.values():
            logsum = float("-inf")
            if hmm.emi[j,o]!=float("-inf"):
                for j in hmm.states.values():
                    if hmm.trans[k,j]!=float("-inf"):
                        logsum = LOGSUM(logsum, B[j, n+1]+hmm.trans[k,j]+hmm.emi[j,o])
            B[k,n] = logsum

    # Posterior decoding:
    M = A+B

    z = ['' for i in range(len(seq))]
    for n in range(N):
        z[n] = hmm.states.keys()[M[:,n].argmax()]

    if len(hmm.states)<10:
        return "".join(z)
    else:
        return z
