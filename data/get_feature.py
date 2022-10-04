
import numpy as np
import math
import itertools
from collections import Counter
from primer3 import calcHairpin


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

# This part is designed for frequency variance
def get_fv(seq):
    base_fre = [ seq.count(base)/len(seq) for base in 'ACGT' ]
    return np.var(base_fre)


def get_cut_feature(seq, lengths, params):
    feature = []
    for cut_length in lengths:
        subs = [ seq[i:i + cut_length] for i in range(0, len(seq), cut_length)]
        subs = subs[:-1]
        for param in params:
            feature_list = [ globals()[ 'get_'+param ](x) for x in subs]
            feature += [ np.min(feature_list), np.max(feature_list), np.var(feature_list) ]
    return feature


# This part is designed for k-mer
def get_kmer(seq, k):
    alphabet = [''.join(e) for e in itertools.product('ACGT', repeat=k)]
    mers = [ seq[i:i+k] for i in range(len(seq)-k+1)]
    kmer = [ mers.count(j)/len(mers) for j in alphabet]
    return kmer


# This part is designed for repeat count
def get_repeat(seq, repeat_length):
    subs = [ seq[i:i + repeat_length] for i in range(len(seq)-repeat_length+1)]
    repeat_times = list(dict(Counter(subs)).values())
    repeat_times = [ x for x in repeat_times if x > 1]
    return np.sum(repeat_times) / (len(seq)-repeat_length+1)


# This part designed for get hairpin structure
def get_hairpin(seq):
    sub_length = 60
    subs = [ seq[i:i + sub_length] for i in range(0, len(seq), sub_length)]
    calcs = [ calcHairpin(x) for x in subs ]
    # ThermoResult(structure_found=True, tm=46.60, dg=-774.84, dh=-25800.00, ds=-80.69)
    hairpins = [ x for x in calcs if x.tm > 45]
    return len(hairpins)


# This part designed for poly structure
def get_poly_base(seq, base):
    cut_length, value = 20, 0.8
    subs = [ seq[i:i + cut_length] for i in range(0, len(seq), cut_length)]
    polys = [x for x in subs if x.count(base) > cut_length*value]
    return len(polys)



def get_feature(seq):
    seq = seq.upper()

    params = [ 'gc', 'ie', 'fe', 'fv']
    feature = [ globals()[ 'get_'+param ](seq) for param in params ]
    feature += get_cut_feature(seq, [20, 100], params)

    Kmer = get_kmer(seq, 3)
    # [6,15,25,50,100]
    Repeat = [get_repeat(seq, i) for i in [3,6,9,12,15,20,25,35,50,75,100]]
    Poly = [get_poly_base(seq, i) for i in 'ACGT']
    feature += Kmer + Repeat + [get_hairpin(seq)] + Poly

    return feature


def feature_extract(seqs):
    features = [ get_feature(seq) for seq in seqs]
    return np.array(features)


if __name__ == '__main__':
    seq = 'AACGACGGCCAGTGCCAAGCTTGCATGGGAACGTGGTTGACCTGCATATTGACGCATACGCTAGCCATGGTCCCAGCCTCCTCGCTGGC'
    seq = seq.upper()
    print(len(seq))
    print( get_cut_feature(seq, [20], ['ie']) )