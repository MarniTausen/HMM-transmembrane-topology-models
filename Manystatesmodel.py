from numpy import matrix
import numpy as np

from HMMdecoding import *

files = ["Dataset160/set160.0.labels.txt", "Dataset160/set160.1.labels.txt", "Dataset160/set160.2.labels.txt", "Dataset160/set160.3.labels.txt",
         "Dataset160/set160.4.labels.txt", "Dataset160/set160.5.labels.txt", "Dataset160/set160.6.labels.txt", "Dataset160/set160.7.labels.txt",
         "Dataset160/set160.8.labels.txt"]

trainingdata = {}
sequences = {}
for i in files:
    for k, v in loadseq(i).items():
        trainingdata[k] = v
        sequences[k] = v[0]

######################################
## States: i (0) iM (1-35) o (36) oM (37-72)


# Count the length of the occurrence of the search term. 
def countCluster(string, s, p):
    c = 0
    for i in string[p:]:
        if i==s: c += 1
        else: break
    return c

def labelMembrane(cl, iM):
    if iM==True:
        cl -= 10
        start = [str(i) for i in range(1,6)]
        end = [str(i) for i in range(31,36)]
        cl -= 5
        core = ['6', '7', '8']+[str(i) for i in range(29-cl,29)]+['29','30']
        return start+core+end
    else:
        cl -= 10
        start = [str(i) for i in range(37,37+5)]
        end = [str(i) for i in range(37+5+25+1,37+5+25+5+1)]
        cl -= 5
        core = ['42', '43', '44']+[str(i) for i in range(66-cl,66)]+['66','67']
        return start+core+end

def TMHmapping(data):
    iM = False

    for keys, values in data.items():
        i = 0
        mapping = []
        while i < len(values[1]):
            if values[1][i] == 'i':
                clen = countCluster(values[1], 'i', i)
                mapping.append(('i', clen))
                i += clen
                continue
            if values[1][i] == 'o':
                clen = countCluster(values[1], 'o', i)
                mapping.append(('o', clen))
                i += clen
                continue
            if values[1][i] == 'M':
                clen = countCluster(values[1], 'M', i)
                mapping.append(('M', clen))
                i += clen
                continue
        result = []
        for i in mapping:
            if i[0]=='i':
                iM = True
                result += ['0']*i[1]
            if i[0]=='o':
                iM = False
                result += ['36']*i[1]
            if i[0]=='M':
                if iM==True:
                    result += labelMembrane(i[1], True)
                else:
                    result += labelMembrane(i[1], False)
        data[keys] = (values[0], result)
    return data

trainingdata = TMHmapping(trainingdata)

def convertBack(seq):
    mapper = {'0': 'i', '36': 'o'}
    r = ""
    for i in seq:
        r += mapper.get(i, 'M')
    return r

def comparing(data):
    iM = False
    result = {}
    for keys, values in data.items():
        i = 0
        mapping = []
        while i < len(values[1]):
            if values[1][i] == 'i':
                clen = countCluster(values[1], 'i', i)
                mapping.append(('i', i+1, clen+i))
                i += clen
                continue
            if values[1][i] == 'o':
                clen = countCluster(values[1], 'o', i)
                mapping.append(('o', i+1, clen+i))
                i += clen
                continue
            if values[1][i] == 'M':
                clen = countCluster(values[1], 'M', i)
                mapping.append(('M', i+1, clen+i))
                i += clen
                continue
        result[keys] = mapping
    return result