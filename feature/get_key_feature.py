
import numpy as np
import math
from collections import Counter



# This part is designed for gc content
def get_gc(seq):
    return (seq.count('C') + seq.count('G')) / len(seq)

# This part is designed for information entropy
def get_ie(seq):
    ie_sum = 0
    for base in 'ACGT':
        if seq.count(base) > 0:
            p_i = seq.count(base) / len(seq)
            ie_sum += p_i * math.log(p_i, 2)
    return -ie_sum

# This part is designed for free energy
def get_fe(seq):
    energy = {'AA':1.9,'AC':1.3,'AG':1.6,'AT':1.5,
              'CA':1.9,'CC':3.1,'CG':3.6,'CT':1.6,
              'GA':1.6,'GC':3.1,'GG':3.1,'GT':1.3,
              'TA':1.5,'TC':1.6,'TG':1.9,'TT':1.9 }
    sum_free = [ energy[ seq[i:i+2] ] for i in range(0, len(seq)-1)]
    return -np.sum(sum_free)


def get_max_GC(seq, length):
    subs = [ seq[i:i + length] for i in range(0, len(seq), length)]
    subs = subs[:-1] if len(subs[:-1]) != length else subs
    value_list = [ get_gc(x) for x in subs]
    return np.max(value_list)

def get_min_IE(seq, length):
    subs = [ seq[i:i + length] for i in range(0, len(seq), length)]
    subs = subs[:-1] if len(subs[:-1]) != length else subs
    value_list = [ get_ie(x) for x in subs]
    return np.min(value_list)


# This part is designed for GGC frequency
def get_GGC(seq, k):
    mers = [ seq[i:i+k] for i in range(len(seq)-k+1)]
    kmer = mers.count('GGC')/len(mers)
    return kmer


# This part is designed for repeat count
def get_repeat(seq, repeat_length):
    subs = [ seq[i:i + repeat_length] for i in range(len(seq)-repeat_length+1)]
    repeat_times = list(dict(Counter(subs)).values())
    repeat_times = [ x for x in repeat_times if x > 1]
    return np.sum(repeat_times) / (len(seq)-repeat_length+1)


def get_feature(seq):
    seq = seq.upper()
    feature = [get_fe(seq), get_GGC(seq, 3), get_repeat(seq, 6), 
               get_min_IE(seq, 100), get_max_GC(seq, 20), get_repeat(seq, 12)]
    return feature


def feature_extract(seqs):
    features = [ get_feature(seq) for seq in seqs]
    return np.array(features)


if __name__ == '__main__':
    seq = 'GGAAgcatCGTGGTTGACCTGCATATTGACGCATACGCTAGCCATGGTCCCAGCCTCCTCGCTGGC'
    print(len(seq))
    print( get_feature(seq) )