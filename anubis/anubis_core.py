#!/usr/bin/env python3

#############################################################
#
# anubis.py
#
# Author : Miravet-Verde, Samuel
#
#############################################################

#####################
#   PACKAGES LOAD   #
#####################

import os
import re
import sys
import copy
import regex
import scipy
import pickle
import random
import itertools
import numpy as np
import pandas as pd
import ruptures as rpt

from pylab import cm
import seaborn as sns
sns.set_style("white")
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from Bio import SeqIO
from Bio.SeqUtils import GC
from itertools import groupby, product
from operator import itemgetter
from collections import Counter, OrderedDict
from scipy.stats import poisson, binned_statistic, norm, lognorm
from scipy.stats.stats import pearsonr
from scipy.optimize import curve_fit

from joblib import Parallel, delayed
import multiprocessing

# METHODS IMPORT

#####################
# GENERAL FUNCTIONS #
#####################

# FILE HANDLING
def load_ins_file(inFile, lg=816394, header=False):
    """
    Load an insertions file (position\treads) from inFile,
    gl corresponds to gene length and it is required to adjust the possible
    genome positions that passed the size of the genome.
    """
    result = {}
    if inFile.endswith('.wig'):
        print('Wig not supported, interpreted as position-reads skipping first line')
        header = True
    with open(inFile, 'rU') as fi:
        for line in fi:
            if header:
                header=False
            else:
                line = line.strip().split()
                if len(line)==2:
                    pos, reads = int(line[0]), float(line[1])
                else:
                    pos, reads = int(line[1]), float(line[2])
                if pos>lg:
                    pos -= lg
                    if pos in result.keys():
                        result[pos]+=reads
                    else:
                        result[pos]=reads
                else:
                    result[pos]=reads
    return result

def df2lol(df):
    """
    Transform a df into a list of list, where each list was
    originally a column.
    """
    lol = []
    for column in df:
        lol.append(list(df[column]))
    return lol

def dict2array(dic):
    if type(array)==dict:
        res = np.array([[k, v] for k, v in array.items()])
    elif type(array[0])==int:
        res = np.array([[k, 0] for k in array])
    return res

def load_goldset_from_file(inFile, index=1):
    """
    Try to load a dictionary with genes from the second sheet template
    and return that dictionary. {} otherwise.
    """
    if inFile.endswith('.csv') or inFile.endswith('.txt'):
        goldset = pd.read_csv(inFile, sep='\t')
    elif inFile.endswith('.xlsx') or inFile.endswith('.xls'):
        goldset = pd.read_excel(inFile, dtype={'gene':str,'class':str}, sheet_name=index)
    else:
        sys.exit("No compatible goldset file provided.")
    return goldset.set_index(goldset.columns[0]).to_dict()[goldset.columns[1]]

def load_collection_from_file(inFile, dtypes, sep='\t'):
    """
    Load the collection from a template file, autodetect if xlsx or csv
    sep represents the separator, please set it as tab in your definition file
    return a dataframe and a list of lists that can be processed in the same way than the package by default
    """
    if inFile.endswith('.csv'):
        df = pd.read_csv(inFile, sep, dtype=dtypes)
        goldset = {}
    elif inFile.endswith('.xlsx') or inFile.endswith('.xls'):
        df = pd.read_excel(inFile, dtype=dtypes)
        goldset = load_goldset_from_file(inFile)
    else:
        sys.exit("No compatible file provided.")
    df  = df.set_index('identifier', drop=False)
    lol = df2lol(df)

    # Remap annotation for non matchiung genome-annotation cases and modify the dataframe
    lol[4] = [m if n in [0, '0'] else n for m, n in zip(lol[3], lol[4])]
    df[df.columns[4]]=lol[4]

    # Be aware we return a dictionary out of the goldset dataframe
    return df, lol, goldset

def prepare_lol(lol):
    """ Prepare list of lists in the collection parser """
    expected_length = max([len(l) for l in lol])
    new_lol = []
    for l in lol:
        if type(l)==str:
            new_lol.append([l]*expected_length)
        elif type(l)==list:
            if len(l)==expected_length:
                new_lol.append(list(l))
            elif len(l)==1:
                new_lol.append(l*expected_length)
            else:
                if len(l) not in [0, 1, expected_length]:
                    sys.exit("No matching length in the list of attributes passed to the collection object")
                else:
                    new_lol.append([0]*expected_length)

    # Repair lol[4] (annotation)
    new_lol[4] = [m if n in [0, '0'] else n for m, n in zip(new_lol[3], new_lol[4])]
    return new_lol

def load_genome_sequence(genome):
    """ Load a genbank/fasta file and return the genome as string """
    # Determine the file type:
    accepted = True
    if genome.endswith('gb') or genome.endswith('gbk') or genome.endswith('genbank'):
        tipo = 'genbank'
    elif genome.endswith('fa') or genome.endswith('fna') or genome.endswith('fasta'):
        tipo = 'fasta'
    else:
        flagpproach = False
        sys.exit('Genome not in accepted format. Use fasta (.fa, .fna, .fasta) or genbank (.gb, .gbk, .genbank).')

    if accepted:
        handle = open(genome, 'rU')
        for record in SeqIO.parse(handle, tipo):
            return str(record.seq)
        handle.close()

def load_annotation_from_genbank(genome, feature_types=['CDS', 'rRNA', 'tRNA', 'ncRNA']):
    """
    Load an genbank file and extract the feature_types in form
    Returns a dictionary {gene:[int(min(st,en)), int(max(st, en)), strand]}
    """
    annotation = {}
    with open(genome, "rU") as input_handle:
        for record in SeqIO.parse(input_handle, "genbank"):
            for feat in record.features:
                if feat.type in feature_types:
                    annotation[feat.qualifiers['locus_tag'][0]] = [int(feat.location.start)+1, int(feat.location.end), '+' if feat.location.strand==1 else '-']
                    # +1 required for the start, if not included the returned positions does not correspond to the annotation.
                    # Biopython changes the notation to base 0. This project works in base 1.
    return annotation

def load_annotation_from_file(genome):
    """
    Load an annotation file in form gene\tstart\tend\tstrand(+|-)\n
    Returns a dictionary {gene:[int(min(st,en)), int(max(st, en)), strand]}
    """
    annotation = {}

    if os.path.isfile(genome):
        with open(genome) as fi:
            for line in fi:
                line = line.strip().split()
                ide  = line[0]
                if len(line[1:]) == 2:
                    st, en = sorted([int(x) for x in line[1:]])
                    l      = [st, en]
                else:
                    st, en = sorted([int(x) for x in line[1:] if x not in ['+', '-']])
                    strand = [str(x) for x in line[1:] if x in ['+', '-']]
                    l      = [st, en] + strand
                annotation[ide] = l
        return annotation
    else:
        sys.exit('Annotation file not found in '+genome+'\n')

def dict2df(dictionary, genome_size=False):
    """
    Transform a dictionary to a dataframe adding 0s to positions
    not included in dictionary with size <genome_size>
    """
    keys = sorted(dictionary.keys())
    if genome_size:
        dictionary = {k:[k, dictionary[k]] if k in keys else [k, 0.0] for k in range(1, genome_size+1)}
    else:
        dictionary = {k:[k, dictionary[k]] for k in sorted(keys)}
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['position', 'reads'])
    return df

def revcomp(seq):
    d = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    return ''.join([d[i] for i in seq])[::-1]

def subseq(sequence, st=None, en=None):
    """
    IMPORTANT: 1-base system, it will be transalted to 0 by the program
    Extract from sequence the range sequence[st:en] or sequence[i-size/2, i+size/2]) 
    Examples:
    st=4, en=7
    1-2-3-4-5-6-7-8
          |------

    st=8, en=3 (treated as circular)
    1-2-3-4-5-6-7-8
    -----         |
    """
    l = len(sequence)
    if st==0:
        st=l
    if en==0:
        en=l
    if en<st:
        en+=l
    if st>l or en>l:
        sequence += sequence

    return sequence[st-1:en]

# PROCESSING FUNCTIONS

def load_dataset(path):
    """ Pickle loader """
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def twoD2dict(a):
    return dict(a)

def dict2twoD(d):
    return np.array([[key, d.profile[key]] for key in d.profile.keys()])


def list2ranges(lista, annotate='ig', min_size=0, side_removal=0):
    """
    Transforms array type <lista> into a list of ranges:
        [1,2,3,4,5, 20,21,22] --> [[1,5], [20,22]]
    By default transforms the ranges to annotation,
    if annotate==str, uses that extension as identifier
    min_size allows to select size base annotations
    side_removal allows to remove <side_removal> positions from
    the edges of the annotations (150 --> 150 each side, 300 in total
    """
    sublista = [list(map(itemgetter(1), g)) for _, g in groupby(enumerate(sorted(lista)), lambda i_x:i_x[0]-i_x[1])] # [1,2,3,4,5,20,21,22] --> [[1,2,3,4,5], [20,21, 22]]]
    sublista = [[min(f), max(f)] for f in sublista]
    if min_size or side_removal:
        sublista = [[i[0]+side_removal, i[1]-side_removal] for i in sublista if i[1]-i[0]+1-side_removal*2>=min_size]
    if annotate:
        if type(annotate)!=str:
            annotate='ig'
        n = len(sublista)
        expected_pad = len(str(n))
        ides = [annotate+str(i).zfill(expected_pad) for i in range(1, n+1)]
        sublista = {k:v+['+'] for k, v in zip(ides, sublista)}
    return sublista

def perc2index(perc, l):
    return int(perc/100.0*l)

def index2perc(index, l):
    return 100.0*index/l

def lists2list(nested_lists):
    nested_list = set(itertools.chain(*nested_lists))
    return nested_list

def predict_N_C_terminal(annotation, data, window_size=50, Nmax=10, Cmax=10):
    """
    input = [st, en, strand]
    Allows to predict for a gene the percentage required to remove from each end.
    N and Cmax determines which is the maximum percentage that will be removed from each end.
    """

    if type(annotation)==dict:
        try:
            _z = len(data.zreads)
        except:
            sys.exit('No data')
        roll = windows(data.zreads, window_size, 'sum', circular=True)
        return {k:predict_N_C_terminal(v, roll, window_size, Nmax, Cmax) for k, v in annotation.items()}
    elif type(annotation)==list:
        st, en, strand = annotation
        l = en-st+1

        subroll = list(data)[st-1:en]
        LR = [Nmax, Cmax]

        if strand=='-':
            LR = LR[::-1]
        LR[1] = 100-LR[1]

        nroll = subroll[:perc2index(LR[0], l)]
        nindex = nroll.index(min(nroll))
        N = index2perc(nindex, l)

        croll = subroll[perc2index(LR[1], l):]
        croll = croll[::-1]
        cindex = croll.index(min(croll))
        C = index2perc(cindex, l)

        if strand=='-':
            return [C,N]
        else:
            return [N,C]   # N=C if strand=neg

    else:
        print('NO PREDICTED')
        return [5,5]

def process_annotation(annotation, Nterminal=5, Cterminal=5, predict=False, data=False, window_size=50, Nmax=10, Cmax=10):
    # IF DEFAULT UPDATED HERE, UPDATE dataset.gene_filter() AS WELL!
    """
    Process annotation {gene:[st, en, strand]} or a list [st, en, strand]
    to remove the <Nterminal>% of the Nterminal end the same for Cterminal using
    <Cterminal>. The strand is required to define what is the N and the C
    terminal.
    <predict> allows to predict for each gene the percentage required to remove [CALLED AUTO IN THE GENERAL PROJECT]
    <Nmax>, <Cmax> max percentage to remove from Nterminal/Cterminal
    <
    """
    if type(annotation)==dict:
        new_dict={k:process_annotation(v, Nterminal, Cterminal, predict, data, window_size, Nmax, Cmax) for k, v in annotation.items()}    # Be aware is recursive!
        return new_dict
    elif type(annotation)==list and type(annotation[0])==int and type(annotation[1])==int and type(annotation[2])==str:
        st, en, strand = sorted(annotation[:2])+[annotation[2]]
        l = en-st+1
        if predict:
            Nterminal, Cterminal = predict_N_C_terminal([st, en, strand], data, window_size, Nmax, Cmax)
        ntoremove = int(l*0.01*Nterminal)
        ctoremove = int(l*0.01*Cterminal)
        if strand=='+':
            new_st = st+ntoremove
            new_en = en-ctoremove
        else:
            new_en = en-ntoremove
            new_st = st+ctoremove
        if st<en:
            return [new_st, new_en, strand]
        else:
            sys.exit('Too small annotation', [st, en, strand])
    else:
        sys.exit('Wrongly formated annotation, pass dictionary or list')

def positions_covered_by_annotation(annotation):
    """
    Return an array with the positions covered by an annotation ([st, en, strand])
    or a dictionary of them.
    """
    if type(annotation)==dict:
        covered = []
        for k, v in annotation.items():
            covered += positions_covered_by_annotation(v)
        return sorted(set(covered))
    elif type(annotation)==list and type(annotation[0])==int and type(annotation[1])==int and type(annotation[2])==str:
        st, en = sorted(annotation[:2])
        return range(st, en+1)
    else:
        sys.exit('Wrongly formated annotation, pass dictionary or list')

def merge_replicates(samples, metric='average'):
    """
    Merge replicates in samples computing the <metric>.
    Metrics accepted are 'sum' or 'average'
    Returns a single sample with the shared information between replicates
    """
    if metric in ['sum', 'average']:
        new_reads = []
        for s in samples:
            if len(new_reads)==0:
                new_reads = s.zreads+0 # sum 0 to keep s.zreads safe
            else:
                new_reads+= s.zreads
        if metric=='average':
            new_reads = new_reads/float(len(samples))
    else:
        sys.exit('Metric not found.')

    return new_reads



      # (self         , location       , genome         , log=None       , genome_sequence=None,
      #   308                  goldset={}   , annotation=None, profile=None   ,
      #  309                  identifier='', time=0.0       , passage=0.0    , dilution=0.0   , replicate=0.0 , doubling_time=0.0, condition=0.0, treatment=0.0)

    #new_dataset = dataset(location=

#########
# PREPROCESSING FUNCTIONS: sample level
#########

def dict_filter(your_profile, min_value, max_value=False):
    """
    Return filtered dictionary
    by value of reads, subset positions with >= value
    if max_value reads value<=reads<=max_value is returned
    """
    if max_value:
        return {k:v for k, v in your_profile.items() if min_value<=v<=max_value}
    else:
        return {k:v for k, v in your_profile.items() if min_value<=v}

def gene_threshold(your_dataset, percentile=95, list_of_genes=None, auto=False, Nterminal=10, Cterminal=10, window_size=50, Nmax=10, Cmax=10):
    """
    Filter the dataset based on the percentile of reads mapping to a set of
    known E genes. Ideas behind this filter implie that a gold set of E genes
    will present mostly artefactual/background insertions and every position
    with similar conditions cannot be trusted.
    TO BE AWARE: usually we remove the 5% of N and C terminal regions of the
    gene, in this case we remove the 10% of each by default to be sure we do
    not include NE positions.
    """
    if not list_of_genes:
        if len(your_dataset.goldE)==0:
            sys.exit('No list of genes available. Either define a goldset or pass it through "list_of_genes" argument')
        else:
            list_of_genes = [i for i in your_dataset.goldE]  # Hardcoded way to copy the list
    # Ensure the genes are available and extract their annotation
    list_of_genes = set(list_of_genes)
    all_genes     = set(your_dataset.annotation.keys())
    intersection  = list_of_genes.intersection(all_genes)
    if intersection==0:
        sys.exit('List of genes provided not included in annotation.')
    genes_to_use = {k:your_dataset.annotation[k] for k in intersection}

    if auto:
        roll = windows(your_dataset.zreads, window_size, 'sum', circular=True)
    else:
        roll = False

    genes_to_use = process_annotation(genes_to_use, Nterminal=Nterminal, Cterminal=Cterminal, predict=auto, data=roll, window_size=window_size, Nmax=Nmax, Cmax=Cmax)
    # Extract the positions in the genes
    indices   = np.array(positions_covered_by_annotation(genes_to_use))-1     # -1 require to set it to 0 base
    covered   = np.take(your_dataset.zreads, indices)                                 # Be aware we use the reads with 0 values, required for next step
    threshold = np.percentile(covered[covered>0], percentile)                 # In here we remove 0 again to calculate the percentile
    # Filter:
    return threshold


#########
# BY functions
#########

def return_I(values):
    """ Return the number of insertions in values """
    values = np.array(values)
    return float(len(values[values>0.0]))

def return_R(values):
    """ Return the total reads in values """
    return float(sum(values))

def replace_zeroes(values):
    min_nonzero = np.min(values[np.nonzero(values)])
    values[values == 0] = min_nonzero
    return values

def return_dens(values):
    return return_I(values)/len(values)

def return_RI_old(values):
    """
    Return the number of reads per insertion in values
    values is expected to be sample.zreads
    """
    _i = return_I(values)
    _r = return_R(values)
    if _i==0:
        return 0.0
    else:
        return _r/_i

def return_RI(values):
    """
    Return the number of reads per insertion in values
    values is expected to be sample.zreads
    """
    _i = return_I(values)
    _r = return_R(values)
    if _i==0:
        return 0.0
    else:
        return _r*_i/len(values)

def return_CPM(values, total_reads):
    """ Returns CPM for values """
    return 1e6*return_R(values)/total_reads

def return_RPKM(values, total_reads):
    """ Returns RPKM for values """
    return return_R(values)/((len(values)/1000.0)*(total_reads/1e6))

def metric_for_annotations(array, annotation, metric, total_reads=0):
    """
    Function to compute metrics over annotation dictionary
    - Ignore allows to pass a list of ranges to ignore,
    useful to deal with repeated regions in the annotatiom
    """

    accepted_metrics = set(['sum', 'std', 'mean', 'median', 'count', 'L', 'min', 'max', 'I', 'dens', 'R', 'RI', 'CPM', 'RPKM'])
    if metric not in accepted_metrics:
        sys.exit(metric+' not accepted. Please provide one of the following:\n- '+'\n- '.join(accepted_metrics))

    specific_metrics = {'sum':sum, 'std':np.std, 'mean':np.mean, 'median':np.median,
                        'count':len, 'L':len, 'min':min, 'max':max,
                        'I':return_I, 'dens':return_dens,
                        'R':return_R, 'RI':return_RI,
                        'CPM':return_CPM, 'RPKM':return_RPKM}

    if metric=='RPKM' or metric=='CPM':
        if total_reads==0:
            total_reads = sum(array)
        results = {k:specific_metrics[metric](array[v[0]-1:v[1]], total_reads=total_reads) for k, v in annotation.items()}
    else:
        results = {k:specific_metrics[metric](array[v[0]-1:v[1]]) for k, v in annotation.items()}
    return results

def ignore_metric_for_annotations(narray, annotation, metric, total_reads=0, ignore=False, gene_len_thr=12, ignore_len_thr=100):
    """
    Function to compute metrics over annotation dictionary
    - Ignore allows to pass a list of ranges to ignore,
    useful to deal with repeated regions in the annotatiom

    only tested for I, R and dens
    """
    array = narray.copy()
    accepted_metrics = set(['sum', 'std', 'mean', 'median', 'count', 'L', 'min', 'max', 'I', 'dens', 'R', 'RI', 'CPM', 'RPKM'])
    if metric not in accepted_metrics:
        sys.exit(metric+' not accepted. Please provide one of the following:\n- '+'\n- '.join(accepted_metrics))

    specific_metrics = {'sum':sum, 'std':np.std, 'mean':np.mean, 'median':np.median,
                        'count':len, 'L':len, 'min':min, 'max':max,
                        'I':return_I, 'dens':return_dens,
                        'R':return_R, 'RI':return_RI,
                        'CPM':return_CPM, 'RPKM':return_RPKM}

    if metric=='RPKM' or metric=='CPM':
        if total_reads==0:
            total_reads = sum(array)
        results = {k:specific_metrics[metric](array[v[0]-1:v[1]], total_reads=total_reads) for k, v in annotation.items()}
    else:
        if ignore:
            for i in ignore:
                array[i-1] = -1
        results = {}
        for k, v in annotation.items():
            subarray = array[v[0]-1:v[1]]
            subarray = subarray[subarray>=0]
            if len(subarray)>=gene_len_thr:
                if len(subarray)!=0:
                    results[k] = specific_metrics[metric](subarray)
                else:
                    results[k] = 0.0
    return results

def windows(array, window_size=False, bins=False, metric='mean', circular=True, overlapping=True):
    """
    Computes the windows for array with window size.
    If circular, the array is considered to be circular

    If overlapping==False, this function produces a bin profile

    bins= represent the number of portions of splitted data (if 100, the profile is separated in 100
    non-overlapping subarray.
    window_size= how many bases each bin has to cover
    """
    accepted_metrics = set(['sum', 'std', 'mean', 'median', 'count', 'L', 'min', 'max', 'I', 'dens', 'R', 'RI', 'CPM', 'RPKM'])
    if type(array)!=np.ndarray:
        array = np.array(array)
    array = array.astype('float')

    if not window_size and not bins:
        sys.exit('At least one of the following arguments is required: window_size or bin')
    if metric not in accepted_metrics:
        sys.exit(metric+' not accepted. Please provide one of the following:\n- '+'\n- '.join(accepted_metrics))
    if metric=='L':
        metric='count'

    if metric in ['dens', 'RI', 'CPM', 'RPKM']:
        original_array = array.copy() # Required to work with recursion

    if circular and overlapping:
        atail = array[-int(window_size/2.0):]
        ahead = array[:int((window_size-1)/2.0)]
        array = np.concatenate((atail,array,ahead))

    if metric=='I' or metric=='dens':
        nz_array = array.copy()
        if overlapping:
            nz_array[nz_array==0.0]='nan'
        else:
            nz_array[nz_array>0.0]=1.0  # Binary array

    if overlapping:
        if metric=='I':
            roll = pd.Series(np.array(nz_array)).rolling(window_size, center=True)
        else:
            if metric not in ['RI', 'RPKM', 'CPM']:
                roll = pd.Series(np.array(array)).rolling(window_size, center=True)

        if metric in ['sum', 'R']:
            roll = roll.sum()
        elif metric=='std':
            roll = roll.std()
        elif metric=='mean':
            roll = roll.mean()
        elif metric=='median':
            roll = roll.median()
        elif metric in 'count':
            roll = roll.count()[len(ahead):len(array)-len(atail)]   # Required to correct the tails
        elif metric=='min':
            roll = roll.min()
        elif metric=='max':
            roll = roll.max()
        elif metric=='I':
            # This is a nonzero_count function as we change 0-->NaN in array variable, check up. 
            roll = roll.count()[len(ahead):len(array)-len(atail)]   # Required to correct the tails
        elif metric=='dens':
            _is  = windows(original_array, window_size=window_size, bins=bins, metric='I', circular=circular, overlapping=overlapping)
            roll = _is/window_size
        elif metric=='RI':
            _is  = windows(original_array, window_size=window_size, bins=bins, metric='I', circular=circular, overlapping=overlapping)
            _rs  = windows(original_array, window_size=window_size, bins=bins, metric='R', circular=circular, overlapping=overlapping)
            roll = _rs*replace_zeroes(_is)
        elif metric=='CPM':
            roll = windows(original_array, window_size=window_size, bins=bins, metric='R', circular=circular, overlapping=overlapping)
            roll = (roll/sum(original_array))*1e6
        elif metric=='RPKM':
            roll = windows(original_array, window_size=window_size, bins=bins, metric='R', circular=circular, overlapping=overlapping)
            roll = roll/((window_size/1000.0)*(sum(original_array)/1e6))
        if metric not in ['dens', 'RI', 'RPKM', 'CPM']:
            roll.dropna(inplace=True)    # Remove tails

    else:
        if type(bins)!=int:
            bins = int(float(len(array))/window_size)

        if bins and not window_size:
            window_size = int(len(array)/bins)

        if metric=='R':
            roll = binned_statistic(np.arange(len(array)), array, bins=bins, statistic='sum')[0]
        elif metric=='I':
            roll = binned_statistic(np.arange(len(nz_array)), nz_array, bins=bins, statistic='sum')[0]
        elif metric=='dens':
            roll = binned_statistic(np.arange(len(nz_array)), nz_array, bins=bins, statistic='sum')[0]
            roll/=window_size
        elif metric=='RI':
            _is  = windows(original_array, window_size=window_size, bins=bins, metric='I', circular=circular, overlapping=overlapping)
            _rs  = windows(original_array, window_size=window_size, bins=bins, metric='R', circular=circular, overlapping=overlapping)
            roll = _rs*_is
        elif metric=='CPM':
            roll = windows(original_array, window_size=window_size, bins=bins, metric='R', circular=circular, overlapping=overlapping)
            roll = (roll/sum(original_array))*1e6
        elif metric=='RPKM':
            roll = windows(original_array, window_size=window_size, bins=bins, metric='R', circular=circular, overlapping=overlapping)
            roll = roll/((window_size/1000.0)*(sum(original_array)/1e6))
        else:
            roll = binned_statistic(np.arange(len(array)), array, bins=bins, statistic=metric)[0]

    return np.array(roll)


def match_substring(s, genome, model, mismatches):
    return len(regex.findall("("+s+"|"+revcomp(s)+"){"+model+"<="+str(mismatches)+"}", genome, overlapped=True))

def duplicated_windows(genome, subsequence=False, window_size=False, bins=False, mismatches=1, circular=True, overlapping=True, model='s'):
    """
    Model defines if substitutions (s) or indels+substitutions (e)
    Not very efficient performance even with the parallelization.
    """

    if not window_size and not bins:
        sys.exit('At least one of the following arguments is required: window_size or bin')

    if circular and overlapping:
        atail  = genome[-int(window_size/2.0):]
        ahead  = genome[:int((window_size-1)/2.0)]
        genome = atail+genome+ahead

    if subsequence:
        windows_map = [genome[i:i+window_size] for i in range(len(subsequence)-window_size+1)]
    else:
        windows_map = [genome[i:i+window_size] for i in range(len(genome)-window_size+1)]
    num_cores = multiprocessing.cpu_count()-1
    unique_windows_map = list(set(windows_map))
    results = Parallel(n_jobs=num_cores)(delayed(match_substring)(s, genome, model, mismatches) for s in unique_windows_map)

    counts = {s:c for s, c in  zip(unique_windows_map, results)}

    return counts

    # windows_map = {i:windows_map[i-1] for i in range(1, l+1)}

    # genome_windows = list(set([genome[i:i+window_size] for i in range(len(genome)-window_size+1)]))
    # genome_windows += [revcomp(s) for s in genome_windows]
    # genome_windows = set(genome_windows)
    # pattern = "("+'|'.join(genome_windows)+"){e<=1}"

    # for k, v in windows_map.items():
    #     pattern =  "("+v+"){"+model+"<=1}"
    #     repeated = len(regex.findall(v, genome, overlapped=True))+len(regex.findall(v, rcgenome, overlapped=True))
    #     windows_map[k] = [v, repeated]


#########
# NORMALIZATION/STANDARDIZATION METHODS
# Collection of linear and non-linear normalization methods
# you can add new arguments adding them to norm_methods dictionary
#########

def linear_stats(x, y):
    """ Return the linear regression stats for two distributions x and y """

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    x_0, x_1 = min(x), max(x)
    y_0 = min(y)
    y_1 = slope*(x_1-x_0)+y_0
    return slope, intercept, r_value, p_value, std_err, x_0, x_1, y_0, y_1


def zscore(value, distribution=None, multiplying_factor=1.0):
    """
    Apply a z standarization to the value.
    Value can be another distribution.
    """
    value = np.array([float(i) for i in value])
    if distribution:
        return multiplying_factor*(np.array(value)-np.mean(np.array(distribution)))/np.std(np.array(distribution))
    else:
        return multiplying_factor*(np.array(value)-np.mean(value))/np.std(value)

def minmax(value, distribution=None, multiplying_factor=1.0):
    """
    Apply a min max standarization to the value.
    Value can be another distribution
    """
    value = np.array([float(i) for i in value])
    if distribution:
        return multiplying_factor*(np.array(value)-min(distribution))/(max(distribution)-min(distribution))
    else:
        return multiplying_factor*(np.array(value)-min(value))/(max(value)-min(value))

########
# MOTIF SEARCH
#######

def translate_coordinates(gene_annotation, domain_annotation):
    """ Adjust hmmscan coordinates to the reference genome """
    gst, gen, gstrand = gene_annotation
    dst, den = domain_annotation
    dnal = float(gen-gst+1)

    if dnal%3!=0.0:
        sys.exit('Wrongly formated annotation.')
    else:
        protl=dnal/3.0

    if gstrand=='+':
        nwst = dst*3+gst
        if den>=protl:
            nwen = gen
        else:
            nwen = den*3+gst
    else:
        if den>=protl:
            nwst = gst
        else:
            nwst = gen-den*3
        nwen = gen-dst*3

    if not gst<=nwst<=nwen<=gen:
        if nwst<gst:
            nwst=gst
        if nwen>gen:
            nwen=gen

    return [nwst, nwen]


def process_domtblout(inFile, annotation, evalue=False, silent=True):
    """
    Given a file generated by hmmscan (integrated in hmmer),
    return a dictionary {gene:[[domain_name, domain start, domain end]]}}

    coordinates is a list of list in order to efficiently
    manage if a domain appears more than once in a sequence,

    evalue defines the threshold to trust a domain, 0.1 by default.
    """
    seqides = set(annotation.keys())
    domains = {}
    with open(inFile, 'rU') as fi:
        for line in fi:
            if line[0]!='#':
                line = line.strip().split()
                # Extract the 'envelope coordinates' more information in page 26
                # of http://eddylab.org/software/hmmer3/3.1b2/Userguide.pdf

                line[6] = float(line[6])
                # Check evalue and proceed
                if not evalue:
                    _flag=True
                else:
                    if line[6]<=evalue:
                        _flag=True
                    else:
                        _flag=False
                if _flag:
                    seqide = line[3]
                    domide = line[1]+'|'+line[0]
                    st, en = sorted([int(i) for i in line[19:21]])

                    if seqide in seqides:
                        ann = [domide]+translate_coordinates(annotation[seqide], [st, en])+[line[6]]
                        if seqide in domains:
                            domains[seqide]+=[ann]
                        else:
                            domains[seqide] =[ann]
                    else:
                        if not silent:
                            print(seqide+' not found in annotation.')
    return {k:sorted(v, key=itemgetter(1)) for k, v in domains.items()}

####
# TERMINA
####
# [metric](array[v[0]-1:v[1]]

def metagene(your_dataset, metric='dens', annotation=False, step=100, extend=0, extend_mode='perc'):
    """ extend_mode can be 'perc' or 'base' """

    if not annotation:
        annotation=your_dataset.annotation
    metaprofile = []
    for k, v in annotation.items():
        st, en, strand = sorted(v[:2])+[v[-1]]

        if extend>0:
            if extend_mode=='perc':
                l = abs(en-st)+1
                extra = int(l*(extend/100))
        else:
            extra = 1*extend

        st = st-1-extra
        en = en+extra
        # print(k, v, st, en)

        if st<0:
            a = np.array(list(your_dataset.zreads[st:])+list(your_dataset.zreads[0:en+1]))
        elif en>your_dataset.genome_length:
            a = np.array(list(your_dataset.zreads[st:your_dataset.genome_length])+list(your_dataset.zreads[0:en-your_dataset.genome_length+1]))
        else:
            a = your_dataset.zreads[st:en]


        if v[-1]=='-':
            metaprofile.append(windows(a, bins=step, metric=metric, circular=False, overlapping=False)[::-1])
        else:
            metaprofile.append(windows(a, bins=step, metric=metric, circular=False, overlapping=False))
    return metaprofile



####
# Differential comparisons
####



def differential(your_dataset, other_dataset=None,
                 target='genome', background='genome',
                 metric='dens', by='window', normalization='minmax', scaling=False,
                 log2=False,  multiplying_factor=1.0,
                 annotation_dict=None, additional_annotation=None,
                 annotation_shortening=True, shortening_metric='RI', Nterminal=5, Cterminal=5,
                 predict=False,
                 bins=False, window_size=50, Nmax=10, Cmax=10,
                 inplace=True):

    """
    Span and background can be:
        - gene
        - [st, en, strand]
        - genome
    This allows to perform multiple comparisons of genetic elements
    """

    try:
        your_dataset.identifier
        if other_dataset:
            other_dataset.identifier
    except:
        sys.exit('Object provided cannot be interpreted as ANUBIS datasets')

    # Define target and background metrics, minimize recomputations

    da = your_dataset.copy('swallow')

    # Define target





    if other_dataset:
        db = other_dataset.copy('swallow')

    return [da, db]


#    def metric_by(self, metric='all', by='annotation',
#                  normalization='minmax', scaling=False, log2=False, multiplying_factor=1.0,
#                  annotation_dict=None, additional_annotation=None,
#                  annotation_shortening=True, shortening_metric='RI', Nterminal=5, Cterminal=5,
#                  predict=False,
#                  bins=False, window_size=50, Nmax=10, Cmax=10,
#                  inplace=True):


# Plotting multiple

def sort_nicely(l):
    """
    Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort(key=alphanum_key)

def multi_plot_profile(datasets, cmap='mako',
                       gene_id=None, start=None, end=None, log2=True, color='darkcyan', save=False, ylim=False,
                       metric='RI', bins=False, extend='auto',
                       Nterminal=5, Cterminal=5, predict=False, window_size=20, Nmax=10, Cmax=10, show_short=None,
                       show_domains=False, domain_levels='auto', domain_colors='auto', domain_evalue=0.1, show_plot=True,
                       previous_ax=None, _mark_positions=False, __close__=True):
    your_ax = False
    if type(datasets)!=list:
        if type(datasets)==dict:
            ks = list(datasets.keys())
            sort_nicely(ks)
            datasets = [datasets[k] for k in ks]
        elif type(datasets)==collection or type(datasets)==protocol:
            ks = list(datasets.datasets.keys())
            sort_nicely(ks)
            datasets = [datasets.datasets[k] for k in ks]
        else:
            sys.exit('Not accepted object')

    # Set color map
    cmap = cm.get_cmap(cmap, len(datasets))
    colors = [rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    # Run first to set up all labels, domains, etc.
    f, your_ax = datasets[0].plot_profile(gene_id=gene_id, start=start, end=end,
                                          window_size=window_size, log2=log2, color2=colors[0], ylim=ylim,
                                          metric=metric, bins=bins, extend=extend,
                                          Nterminal=Nterminal, Cterminal=Cterminal, predict=predict,
                                          Nmax=Nmax, Cmax=Cmax, show_short=show_short,
                                          show_domains=show_domains, domain_levels=domain_levels,
                                          domain_colors=domain_colors, domain_evalue=domain_evalue, _mark_positions= _mark_positions,
                                          show_insertions=False,
                                          show_plot=False, __close__=False)

    h = [plt.plot([],[], color=c, label=d.identifier.split('/')[-1])[0] for c, d in zip(colors, datasets)]
    lgd = f.legend(handles=h,fontsize=15, loc='center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=5)

    c = 1
    for d in datasets[1:]:
        f, your_ax = d.plot_profile(gene_id=gene_id, start=start, end=end,
                                    window_size=window_size, log2=log2, color2=colors[c], ylim=ylim,
                                    show_insertions=False, show_plot=False, show_short=False,
                                    show_domains=False, _mark_positions=False,
                                    previous_ax=[f, your_ax], __close__=False)
        c+=1
    if save:
        f.savefig(save, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()

def ruptures(your_dataset, gene_id=None, start=None, end=None,
             window_size=10, metric='dens',
             model='rbf', pen=10, min_size='auto',
             save=False):
    """ see https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/ """

    if gene_id:
        st, en, _ = your_dataset.annotation[gene_id]
    elif start and end:
        st, en = [start, end]
    else:
        sys.exit('No annotation provided')

    your_dataset._update_roll_(window_size=window_size, metric='dens', overlapping=True)
    signal = your_dataset._roll_[st:en]

    # detection
    if min_size=='auto':
        min_size=int((en-st+1)*0.1)
        min_size=2
    else:
        if type(min_size)!=int:
            min_size=2
    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    result = algo.predict(pen=pen)

    # display
    rpt.display(signal, result)
    plt.show()

    print(st+np.array(result))

    ignore = list(range(st, result[0]+st))+list(range(st+result[1], en))
    your_dataset.set_domains('', ignore=ignore, prefix='ter')
    your_dataset.plot_profile(gene_id, _mark_positions=st+np.array(result[:-1]),
                              window_size=10, metric='RI', show_domains=True,
                              domain_levels=1, domain_colors=1, save=save)

    print(round(index2perc(result[0], en-st+1), 2), round(100-index2perc(result[1], en-st+1),2))

####
# COVERAGE ESTIMATION
####

def _poisson_event(r, n, approximate=True, norm=True):
    """
    From
    https://math.stackexchange.com/questions/32800/probability-distribution-of-coverage-of-a-set-after-x-independently-randomly
    https://math.stackexchange.com/questions/203491/expected-coverage-after-sampling-with-replacement-k-times
    r = sampling size (e.g. number of cells transformed)
    n = population size (e.g. genome size)
    """
    if approximate:
        val = (1-np.exp(-1*(r/n)))
    else:
        val = (1-((n-1)/n)**r)
    if norm:
        val *= 100
    return val

def _poisson_exponent(e):
    """ Derivation of x/n from expected_coverage """
    return -1*np.log(1-e)

def _poisson_sampling(n, e):
    """ Derivation of x from expected_coverage """
    return n*_poisson_exponent(e)

def _poisson_variance(r, n, norm=True):
    val  = n*(1-(1/n))**r
    val += n**2*(1-(1/n))*(1-(2/n))**r
    val -= n**2*(1-(1/n))**(2*r)
    if norm:
        val = val/n*100
    return val

def expected_coverage(transformed_cells, genome_size, approximate=True, norm=True):
    return _poisson_event(transformed_cells, genome_size, approximate, norm)

def expected_ratio(expected_coverage):
    """ Derivation of x/n from expected_coverage """
    return _poisson_exponent(expected_coverage)

def expected_sampling(genome_size, expected_coverage):
    """ Derivation of x from expected_coverage """
    return _poisson_sampling(genome_size, expected_coverage)

def variance_expected_coverage(transformed_cells, genome_size, norm=True):
    return _poisson_variance(transformed_cells, genome_size, norm)

#####################
#     CLASSES       #
#####################

class dataset(object):
    """ Generic class to define a insertion dataset and the possible modifications """

    # Attributes
    def __init__(self         , location       , genome         , log=None       , genome_sequence=None,
                 goldset={}   , annotation=None, profile=None   , ignore=None    ,
                 identifier='', time=0.0       , passage=0.0    , dilution=0.0   , replicate=0.0 , doubling_time=0.0, condition=0.0, treatment=0.0,
                 domains=False , domains_evalue=0.1):
        """
        Initiation method,
        Preferences:
        genome_sequence (string) > genome (path)
        annotation (dic/path to file) > extract from genome (path) if genbank
        profile (dic={pos:reads}) > location (path to ins file)
        """
        # Define the attributes
        self.location        = str(location)    # File location information
        self.genome          = str(genome)
        if genome_sequence:
            self.genome_sequence = str(genome_sequence)
        else:
            self.genome_sequence = load_genome_sequence(self.genome)
        self.genome_length   = len(self.genome_sequence)
        self.log             = str(log)

        # Experimental data
        self.time          = float(time)
        self.passage       = float(passage)
        self.dilution      = float(dilution)
        self.replicate     = float(replicate)
        self.doubling_time = float(doubling_time)
        self.condition     = str(condition)
        self.treatment     = str(treatment)

        if identifier=='':
            self.identifier = str(location)
        else:
            self.identifier = str(identifier)

        # Annotation information
        if annotation:
            if type(annotation) is dict:
                self.annotation = annotation
            elif type(annotation) is str and annotation!='0':
                self.annotation = load_annotation_from_file(annotation)
            else:
                pass
        else:
            # This will assume that the genome path directs to a genbank file
            if self.genome.endswith('gb') or self.genome.endswith('gbk') or self.genome.endswith('genbank'):
                self.annotation = load_annotation_from_genbank(self.genome)
            else:
                print('No annotation provided.')
        self.original_annotation=False
        self.additional_annotation=False

        # Add goldset if present and positions to ignore
        self.set_goldset(goldset)
        self.set_ignore(ignore)

        # Profile and stats information
        self._empty_profile_ = False
        if not profile:
            profile = load_ins_file(self.location, self.genome_length)
        else:
            if len(profile)==self.genome_length:
                self.zprofile = profile
            else:
                self.zprofile = {k:0.0 for k in range(1, self.genome_length+1)}
                self.zprofile.update(profile)
        self._update_stats_(profile)

        # Annotation based attributes
        self._one_base_annotation_ = False
        self.intergenic = False

        # DOMAINS
        if domains:
            self.domains_location = motifs
            self.domains = process_domtblout(self.motifs_location, self.annotation, evalue=motifs_evalue)
            self._domains_evalue_ = motifs_evalue
        else:
            self.domains_location = False
            self.domains = False
            self._domains_evalue_ = 0.1

    # OBJECT MODIFIERS
    def _update_stats_from_array_(self, new_aprofile):
        """
        Given a 2D array [[pos, reads]],
        update many statistical elements in the dataset object
        """
        # To fasten other processes, also useful to restart if new profile
        self._window_size_     = 0
        self._roll_            = False
        self._window_metric_   = ''
        self._bin_window_size_ = 0
        self._bin_roll_        = False
        self._bin_metric_      = ''
        self.metric            = False
        self.metric_lengths    = False
        self.metric_ides       = False
        self.metric_values     = False
        self.profile_array     = False

        # To speed up copy (only first time)
        self.aprofile   = new_aprofile
        self.apositions = new_aprofile[:,0]
        self.zreads     = new_aprofile[:,1]
        self.zprofile   = dict(new_aprofile)

        _tmparray       = new_aprofile[new_aprofile[:,1]>0]
        self.profile    = dict(_tmparray)
        self.dataframe  = pd.DataFrame({'position':self.apositions, 'reads':self.zreads})
        self.dataframe.index += 1

        self.positions   = _tmparray[:,0]
        self.reads       = _tmparray[:,1]

        _mask            = self.zreads > 0
        self.zpositions  = _mask.astype(np.int)                               # including zeroes if no reads 1 if !=0
        self.density     = np.mean(self.reads>0)
        self.coverage    = 100.0*sum(self.zpositions)/self.genome_length

    def _update_stats_(self, new_profile):
        """
        Given a dictionary {pos:reads},
        update many statistical elements in the dataset object
        """
        # To fasten other processes, also useful to restart if new profile
        self._window_size_     = 0
        self._roll_            = False
        self._window_metric_   = ''
        self._bin_window_size_ = 0
        self._bin_roll_        = False
        self._bin_metric_      = ''
        self.metric            = False
        self.metric_lengths    = False
        self.metric_ides       = False
        self.metric_values     = False
        self.profile_array     = False

        # To speed up copy (only first time)
        if not self._empty_profile_:
            self.apositions = np.array(range(1, self.genome_length+1))
            self._empty_profile_ = {k:0.0 for k in self.apositions}

        self.profile    = new_profile
        self.zprofile   = copy.copy(self._empty_profile_)
        self.zprofile.update(self.profile)
        self.zprofile   = OrderedDict(sorted(self.zprofile.items()))
        self.zreads     = np.array(list(self.zprofile.values()))
        self.aprofile   = np.column_stack((self.apositions, self.zreads))

        # self.zreads     = np.array([self.profile.get(i, 0.0) for i in self.apositions])
        self.dataframe  = pd.DataFrame({'position':self.apositions, 'reads':self.zreads})
        self.dataframe.index += 1
        self.positions   = np.array(sorted(self.profile.keys()))
        self.reads       = np.array(self.zreads[self.zreads>0.0])
        _mask            = self.zreads > 0
        self.zpositions  = _mask.astype(np.int)                               # including zeroes if no reads 1 if !=0
        self.density     = np.mean(self.reads>0)
        self.coverage    = 100.0*len(self.positions)/self.genome_length

    def copy(self, mode='deep'):
        """
        Return a copy of the dataset
        mode:
            deep = deep copy
            swallow = swallow copy
            anything else = new instance of the class
        """
        if mode=='deep':
            return copy.deepcopy(self)
        elif mode=='swallow':
            return copy.copy(self)
        else:
            return dataset(location=self.location    , genome=self.genome              , log=self.log            , genome_sequence=self.genome_sequence,
                           goldset=self.goldset      , annotation=self.annotation      , profile=self.profile    ,
                           identifier=self.identifier, time=self.time                  , passage=self.passage    , dilution=self.dilution              ,
                           replicate=self.replicate  , doubling_time=self.doubling_time, condition=self.condition, treatment=self.treatment)

    def _set_profile_array_(self, by='I', percentile=False):
        # self.profile_array = np.array([int(i) for i in ','.join([','.join([str(k)]*int(v)) for k, v in self.profile.items()]).split(',')])

        if by=='I':
            return self.positions
        elif by=='R':
            if type(self.profile_array)==bool:
                self.profile_array = []
                for k, v in self.profile.items():
                    self.profile_array+=[k]*int(v)
                self.profile_array = np.array(self.profile_array)
            return self.profile_array
        else:
            sys.exit('<by>:'+str(by)+' not accepted. Pass "I" for insertions or "R" for reads')

    def remove_positions(self, positions, inplace=False):
        """ Set to zero the value of read of <positions> """
        positions = set(positions)
        new_profile = {k:v for k,v in self.profile.items() if k not in positions}
        if inplace:
            self._update_stats_(new_profile)
        else:
            filt_dataset = self.copy('swallow')
            filt_dataset._update_stats_(new_profile)
            return filt_dataset

    def set_ignore(self, positions=None):
        """ Set positions to ignore """
        if positions:
            self.ignore = set(positions)
        else:
            self.ignore = positions

    def reduce_old(self, by='I', n=None, percentage=5, lpercentile=False, rpercentile=100, inplace=False):
        """
        Randomly remove <by> (I-insertions, R-reads).
        A <percentage> or a <n> number of <by> to remove
        eg. percentage=5 --> remove 5%, 95% remains
        """
        if lpercentile>=0 and by=='R':
            minv = np.percentile(self.reads, lpercentile)
            maxv = np.percentile(self.reads, rpercentile)
            print(minv, maxv)
            subprof = dict_filter(self.profile, min_value=minv, max_value=maxv)
        else:
            if not n:
                if by=='R':
                    n = sum(self.reads)
                else:
                    n = len(self.positions)
                n *= (100-percentage)/100.0
            if n<1:
                n = 1
            else:
                n = int(n)
            arr = self._set_profile_array_(by=by)
            arr = np.random.choice(arr, n, replace=False)

            if by=='I':
                arr = set(arr)
                subprof = {k:v for k, v in self.profile.items() if k in arr}
            else:
                subprof = dict(Counter(arr))

        if inplace:
            self._update_stats_(subprof)
        else:
            filt_dataset = self.copy('swallow')
            filt_dataset._update_stats_(subprof)
            return filt_dataset

    def reduce(self, by='I', random=0, percentil=0, inplace=False):
        """ """
        pass






    # SETTERS
    def set_goldset(self, goldset):
        """
        Add a goldset passed by a tab delimited file or a dictionary.
         If already defined it will be replaced.
        """
        if type(goldset) is dict:
            self.goldset = goldset
        else:
            self.goldset = load_goldset_from_file(goldset)
        self.goldE       = [k for k, v in self.goldset.items() if v=='E']
        self.goldNE      = [k for k, v in self.goldset.items() if v=='NE']

    def set_domains(self, inFile, ignore=False, prefix='rep'):
        """
        After running:
            hmmpress ./Pfam-A.hmm/data
            hmmscan --domtblout <inFile> Pfam-A.hmm/data path/fasta/file/with/aa/sequences.fa
        Pass output from hmmscan --domtblout to load the annotation
        <inFile> represents the domtblout path
        <evalue> is the minimum threshold the evalue of a motif needs to pass to be considered. 0.1 by default (same than HMMER)

        If you look to set a custom list of motifs, try set_additional_annotation()
        """
        if not ignore:
            if self.domains_location!=inFile:
                self.domains = process_domtblout(inFile, self.annotation)
                self.domains_location = inFile
        else:
            rs = {}
            for k, v in list2ranges(ignore, annotate=prefix).items():
                rs[v[0]] = [k, v[0],v[1], v[1]-v[0]+1]
                rs[v[1]] = [k, v[0],v[1], v[1]-v[0]+1]

            last_rs = {}
            for k,v in self.annotation.items():
                st, en, strand = v
                for i in range(st, en+1):
                    if i in rs:
                        if k in last_rs:
                            last_rs[k][rs[i][0]]=rs[i]
                        else:
                            last_rs[k]={rs[i][0]:rs[i]}

            self.domains = {k:list(v.values()) for k, v in last_rs.items()}

    def set_additional_annotation(self, dictionary):
        """
        Given a dictionary of {identifier:[start, end, strand]}
        uses this additional information for plotting, additional analysis, etc.

        example: you can pass a specific annotation to be plotted together a gene,
        like a motif
        """
        self.additional_annotation = dictionary

    # +7
    def correlation_by_distance(self, other_dataset=None, min_relative_distance=-20, max_relative_distance=20, read_threshold=16):
        """ Evaluates transposon site duplications """
        results = {k:[[],[]] for k in range(min_relative_distance, max_relative_distance+1)}
        dummy = set(self.profile.keys())
        for i in results.keys():
            for k, v in self.profile.items():
                if v>=read_threshold:
                    j = k+i
                    # if j in dummy and self.profile[j]>=read_threshold:
                    results[i][0].append(v)
                    if other_dataset:
                        results[i][1].append(other_dataset.profile.get(j, 0.0))
                    else:
                        results[i][1].append(self.profile.get(j, 0.0))

        # return results
        results = {k:pearsonr(v[0],v[1])[0] if len(v[0])>1 else np.nan for k, v in results.items()}
        return results

    def correct_TSD(self, fw=None, rv=None,
                    tsd_size=7,
                    fullmerge = False,
                    percentile=90, r_thr=0.5, inplace=False):
        """
        Correct for TSD based on the correlation by percentile
        collapses profiles assuming i+tsd_size for positions with with high correlation
        If percentile=='auto', collapse only for the ercentile with correlation > r_thr
        default 90
        """
        if fw and rv:
            dpe = dataset(self.location, genome=self.genome)
            dfw = dataset(fw, genome=self.genome)
            drv = dataset(rv, genome=self.genome)
        else:
            try:
                dpe = dataset(self.location, genome=self.genome)
                dfw = dataset(self.location.replace('.qins', '_fw.qins'), genome=self.genome)
                drv = dataset(self.location.replace('.qins', '_rv.qins'), genome=self.genome)
            except:
                sys.exit('No fw and rv files passed')

        new_d = {k:dfw.profile.get(k, 0.0)+drv.profile.get(k+tsd_size, 0.0) for k in dfw.apositions}
        new_d = {k:v for k, v in new_d.items() if v>0}
        dco = dataset(location='', genome=self.genome, profile=new_d)

        if not fullmerge:
            if percentile=='auto':
                percl = range(0, 91, 10)
                percr = range(10, 101, 10)
                for l, r in zip(percl, percr):
                    sub_d = dfw.filter(filter_by='tails', lpercentile=l, rpercentile=r)
                    sub_othir_d = drv.filter(filter_by='tails', lpercentile=l, rpercentile=r)
                    if r_thr<sub_d.correlation_by_distance(other_dataset=sub_other_d).get(tsd_size, np.nan):
                        percentile=percl

            filtco = dco.filter(filter_by='tails', lpercentile=90, rpercentile=100, inplace=False)
            subpe  = dpe.filter(filter_by='tails', lpercentile=0, rpercentile=percentile-1, inplace=False)
            new_d = {k:subpe.profile.get(k, 0.0)+filtco.profile.get(k, 0.0) for k in dpe.apositions}
            new_d = {k:v for k, v in new_d.items() if v>0}
            dco = dataset(location='', genome=self.genome, profile=new_d)

        if inplace:
            self._update_stats_(new_d)
        else:
            return dco

    # GC CORRECTION
    def GC(self, k=4, color='darkcyan', color2='blueviolet', save=False, show_plot=True,
           annotation_dict=None, prefix='ig', side_removal=150, min_size=100, inplace=False):
        """ <k> represents the number of characters per subsequence """

        kmers = np.array(sorted([''.join(p) for p in product("ACGT", repeat=k)]))
        gccon = np.array([GC(i) for i in kmers])

        # Extact intergenic positions
        intergenic_positions = set(positions_covered_by_annotation(self.intergenic_annotation(annotation_dict=annotation_dict, prefix=prefix, side_removal=side_removal, min_size=min_size, inplace=True)))
        # intergenic_positions = set(positions_covered_by_annotation({i:self.annotation[i] for i in self.goldNE}))

        # Define susbsequences
        k2 = int(k/2)
        if k%2==0:
            l, r = -k2+1, k2
        else:
            l, r = -k2, k2

        # Define subsequences and background model
        intergenic_kmers = {i:subseq(self.genome_sequence, i+l, i+r) for i in intergenic_positions}
        nkmers = Counter([intergenic_kmers[i] for i in intergenic_positions])   # All sites
        nkmers = np.array([nkmers.get(i, 0.0) for i in kmers])

        # Define inserted kmers
        intergenic_inserted = intergenic_positions.intersection(set(self.positions))
        ikmers = Counter([intergenic_kmers[i] for i in intergenic_inserted])    # Inserted
        ikmers = np.array([ikmers.get(i, 0.0) for i in kmers])

        # Normalized by number of times they appear
        normkmers = ikmers/nkmers

        # Plotting:
        plt.figure(figsize=(7,7))
        plt.subplot(2, 2, 1)
        viridis = cm.get_cmap('mako', len(set(gccon)))

        plt.scatter(nkmers, ikmers, color=viridis(gccon/100.0), alpha=0.65)
        _rs = []
        slope, intercept, r_value, p_value, std_err, x_0, x_1, y_0, y_1 = linear_stats(nkmers, ikmers)
        _rs.append([r_value, p_value])
        plt.plot(np.unique(nkmers), np.poly1d(np.polyfit(nkmers, ikmers, 1))(np.unique(nkmers)), linestyle='--', c=color2, label='R2 = {:.3f}'.format(r_value**2))
        plt.title('Available vs. Inserted {}-mers'.format(k))
        plt.xlabel('Nr Available {}-mers'.format(k))
        plt.ylabel('Nr Inserted {}-mers'.format(k))
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.scatter(gccon, normkmers, color=viridis(gccon/100.0), alpha=0.65)
        slope, intercept, r_value, p_value, std_err, x_0, x_1, y_0, y_1 = linear_stats(gccon, normkmers)
        plt.plot(np.unique(gccon), np.poly1d(np.polyfit(gccon, normkmers, 1))(np.unique(gccon)), linestyle='--', c=color2, label='R2 = {:.3f}'.format(r_value**2))
        plt.title('Prob(Ins) vs. GC')
        plt.xlabel('GC content [%]'.format(k))
        plt.ylabel('Prob(Ins) [Nr Inserted {}-mers/Nr Available {}-mers]'.format(k, k))
        plt.legend()

        corrkmers  = normkmers/(slope*gccon+intercept)

        # mini, maxi = min(corrkmers), max(corrkmers)   # To scale the values
        # corrkmers  = (corrkmers-min(corrkmers))/(max(corrkmers)-min(corrkmers))

        plt.subplot(2, 2, 3)
        plt.scatter(gccon, corrkmers, color=viridis(gccon/100.0), alpha=0.65)
        slope, intercept, r_value, p_value, std_err, x_0, x_1, y_0, y_1 = linear_stats(gccon, corrkmers)
        plt.plot(np.unique(gccon), np.poly1d(np.polyfit(gccon, corrkmers, 1))(np.unique(gccon)), linestyle='--', c=color2, label='R2 = {:.3f}'.format(r_value**2))
        plt.title('Prob(Ins) vs. GC')
        plt.xlabel('GC content [%]'.format(k))
        plt.ylabel('Corrected probabilities'.format(k, k))
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.scatter(ikmers, corrkmers*ikmers, color=viridis(gccon/100.0), alpha=0.65)
        slope, intercept, r_value, p_value, std_err, x_0, x_1, y_0, y_1 = linear_stats(ikmers, corrkmers*ikmers)
        _rs.append([r_value, p_value])
        plt.plot(np.unique(ikmers), np.poly1d(np.polyfit(ikmers, corrkmers*ikmers, 1))(np.unique(ikmers)), linestyle='--', c=color2, label='R2 = {:.3f}'.format(r_value**2))
        plt.title('Available vs. Corrected number inserted {}-mer'.format(k))
        plt.xlabel('Nr Inserted {}-mers'.format(k))
        plt.ylabel('Nr Inserted {}-mers corrected'.format(k, k))
        plt.legend()

        plt.tight_layout()

        if save:
            plt.savefig(save, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()

        weightxkmer = {k:v for k, v in zip(kmers, corrkmers)}
        new_profile = {k:v*weightxkmer[subseq(self.genome_sequence, k+l, k+r)] for k, v in self.profile.items()}
        new_ins = [1*weightxkmer[subseq(self.genome_sequence, k+l, k+r)] for k in self.apositions]
        if inplace:
            self._update_stats_(new_profile)
            return _rs
        else:
            return _rs, new_profile, new_ins

    def length(self, n=100, iterations=500, max_size=None, **kwargs):
        metric = ['L']
        if 'metric' in kwargs:
            if type(kwargs.get('metric'))==list:
                metric+=kwargs.get('metric')
            else:
                metric+=[kwargs.get('metric')]
        else:
            metric+=['dens']

        if 'by' not in kwargs or kwargs.get('by')=='annotation':
            self.metric_by(metric=metric, by='annotation',
                           normalization=kwargs.get('normalization', False),
                           scaling=kwargs.get('scaling', False), multiplying_factor=kwargs.get('multiplying_factor', 1.0),
                           log2=kwargs.get('log2', False),
                           annotation_dict=kwargs.get('annotation_dict', None),
                           additional_annotation=kwargs.get('additional_annotation', None),
                           annotation_shortening=kwargs.get('annotation_shortening', True),
                           shortening_metric=kwargs.get('shortening_metric', 'RI'),
                           bins=kwargs.get('bins', False),
                           window_size=kwargs.get('window_size', 50),
                           predict=kwargs.get('predict_shortening', False),
                           Nterminal=kwargs.get('Nterminal', 5), Cterminal=kwargs.get('Cterminal', 5),
                           Nmax=kwargs.get('Nmax', 10), Cmax=kwargs.get('Cmax', 10), inplace=True)

            l = self.metric_values['L']
            selected_metric = [i for i in self.metric_values.keys() if i not in ['ides', 'L']][0]
            m = self.metric_values[selected_metric]

            for mi, ma in [[0,25], [25, 50], [50, 75], [75, 100]]:
                minthr = np.percentile(m, mi)
                maxthr = np.percentile(m, ma)
                sub_m = [np.mean(np.random.choice(l[(minthr<=m) & (m<maxthr)], n)) for i in range(iterations)]
                sns.distplot(sub_m, label='{}%-{}% {}'.format(mi, ma, selected_metric))
            # sns.distplot(l, label='All genes')
            plt.xlabel('Average length')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
        else:
            if not max_size:
                max_size = max(self._single_metric_by('L').values())
            for size in range(1, max_size, int(max_size/100)):
                self.metric_by(metric=metric, by=kwargs.get('by'),
                               normalization=kwargs.get('normalization', False),
                               scaling=kwargs.get('scaling', False), multiplying_factor=kwargs.get('multiplying_factor', 1.0),
                               log2=kwargs.get('log2', False),
                               annotation_dict=kwargs.get('annotation_dict', None),
                               additional_annotation=kwargs.get('additional_annotation', None),
                               annotation_shortening=kwargs.get('annotation_shortening', True),
                               shortening_metric=kwargs.get('shortening_metric', 'RI'),
                               bins=kwargs.get('bins', False),
                               window_size=size,
                               predict=kwargs.get('predict_shortening', False),
                               Nterminal=kwargs.get('Nterminal', 5), Cterminal=kwargs.get('Cterminal', 5),
                               Nmax=kwargs.get('Nmax', 10), Cmax=kwargs.get('Cmax', 10), inplace=True)

                selected_metric = [i for i in self.metric_values.keys() if i not in ['ides', 'L']][0]
                m = self.metric_values[selected_metric]
                sub_m = [np.mean(np.random.choice(m, n)) for i in range(iterations)]
                sns.distplot(sub_m, label='{}'.format(size))
            plt.xlabel('Average {}'.format(selected_metric))
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

    # QC
    def QC(self, dist='norm', color='darkcyan', color2='blueviolet', save=False, show_plot=True):
        """
        dist:
            str or stats.distributions instance
            Distribution or distribution function name. The default is 'norm' for a normal qq-probability plot.
        """
        # Three plots
        # Plotting:
        plt.figure(figsize=(12,4))
        plt.subplot(1, 3, 1)
        sns.distplot(self.reads, color=color)
        plt.title('Non-zero reads distribution')
        plt.xlabel('Read value [counts]')
        plt.ylabel('Frequency')

        plt.subplot(1,3, 2)
        _ = scipy.stats.probplot(self.reads, dist="norm", plot=plt)


        plt.subplot(1, 3, 3)
        x, y = [], []
        for _x, _y in Counter(self.reads).items():
            x.append(_x)
            y.append(_y)
        plt.scatter(x, y, color=color)
        plt.title('Ranked reads')
        plt.xlabel('Read counts')
        plt.ylabel('Read values')

        plt.tight_layout()

        if save:
            plt.savefig(save, bbox_inches='tight')
        if show_plot:
            plt.show()

    # FILTERING FUNCTIONS
    def filter(self, filter_by='genes',
               min_value=16, max_value=False,
               lpercentile=5, rpercentile=95,
               percentile=95, list_of_genes=None, auto=False, Nterminal=10, Cterminal=10, window_size=50, Nmax=10, Cmax=10,
               inplace=False):
        """
        Main filtering function:
            filter_by:
                - reads : min_value (min_value<=reads), max_value (reads<=max_value), default min_value=16
                - tails : lpercentile and rpercentile for reads (lp<=reads<=rp), default 5-95.
                          Similar to TTR (Trimmed Total Reads) in Transit, by default it
                          trims top and bottom 5% of read-counts.
                - genes : Filter the dataset based on the percentile of reads mapping to a set of
                          known E genes. Ideas behind this filter implie that a gold set of E genes
                          will present mostly artefactual/background insertions and every position
                          with similar conditions cannot be trusted.
                          TO BE AWARE: usually we remove the 5% of N and C terminal regions of the
                          gene, in this case we remove the 10% of each by default to be sure we do
                          not include NE positions.
                          defaults: percentile=95, list_of_genes=Egenes, auto=False, Nterminal=10,
                                    Cterminal=10, window_size=50, Nmax=10, Cmax=10
        """
        if type(filter_by)==str:
            filter_by = [filter_by]

        if not inplace:
            filt_dataset = self.copy('swallow')

        for filt in filter_by:
            if filt=='reads':
                minv = min_value
                maxv = max_value
            elif filt=='tails':
                minv = np.percentile(self.reads, lpercentile)
                maxv = np.percentile(self.reads, rpercentile)
            elif filt=='genes':
                minv = gene_threshold(self, percentile=percentile, list_of_genes=list_of_genes, auto=auto, Nterminal=Nterminal, Cterminal=Cterminal, window_size=window_size, Nmax=Nmax, Cmax=Cmax)
                maxv = max_value
            else:
                sys.exit("No filter method allowed, please provide: 'reads', 'genes', 'tails'")

            # Update
            if inplace:
                self._update_stats_(dict_filter(self.profile, min_value=minv, max_value=maxv))
            else:
                filt_dataset._update_stats_(dict_filter(filt_dataset.profile, min_value=minv, max_value=maxv))
        if not inplace:
            return filt_dataset

    # TERMINAL SHORTENING
    def _update_roll_(self, window_size, metric, overlapping=True):
        if overlapping:
            if self._window_size_!=window_size or self._window_metric_!=metric:
                self._window_size_     = window_size
                self._roll_            = windows(self.zreads, window_size, metric=metric, circular=True)
                self._window_metric_   = metric
        else:
            if self._bin_window_size_!=window_size or self._bin_metric_!=metric:
                self._bin_window_size_ = window_size
                self._bin_roll_        = windows(self.zreads, window_size, metric=metric, circular=False, overlapping=False)
                self._bin_metric_      = metric

    def terminal_shortening(self, Nterminal=5, Cterminal=5, predict=False, metric='RI', window_size=50, Nmax=10, Cmax=10, inplace=False, plot=False):
        """
        Shortens the annotation provided by N and C terminal regions as they are
       expected to accumulate more insertions than general regions of the gene.
        Arguments:
            - Nterminal, default 5, % of length to be removed from Nterminal (start codon region)
            - Cterminal, default 5, % of length to be removed from Nterminal (stop codon region)
            - predict, default False. True uses a smoothing function based on windows to remove the first region saturated of insertions
            - window_size. Window size for the smoothing function, default: 50
            - Nmax and Cmax are boundaries that cannot be surpassed when removing % in the predict mode
            - inplace: if True, annotation will be replaced wiht the shortened version.
        """

        if predict:
            self._update_roll_(window_size, metric)

        new_annotation = process_annotation(annotation=self.annotation, Nterminal=Nterminal, Cterminal=Cterminal, predict=predict, data=self._roll_, window_size=window_size, Nmax=Nmax, Cmax=Cmax)
        if inplace:
            self.original_annotation = {k:v for k, v in self.annotation.items()}
            self.annotation = new_annotation
        else:
            return new_annotation

    # METRIC FUNCTIONS
    def _metric_df(self, results, inplace=True, log2=False):
        if type(results)==dict:
            df = pd.DataFrame.from_dict(results)
            df.set_index('ides', inplace=True)
        else:
            df = results

        if log2:
            ldf = df.apply(np.log2)
            ldf.rename(columns={k:'log_'+k for k in ldf.columns}, inplace=True)
            df = df.join(ldf)

        if inplace:
            self.metric = df
        else:
            return df

    def _single_metric_by(self, metric='RI', by='annotation', annotation_dict=None, additional_annotation=None, annotation_shortening=True, shortening_metric='RI', Nterminal=5, Cterminal=5, predict=False, bins=False, window_size=50, Nmax=10, Cmax=10, inplace=False,  gene_len_thr=10):
        """
        Comments:
            - If you select additional annotation and terminal shortening, additional annotation will not be processed. We assume that here you would
            pass annotations like promoters which shortening make no sense.
        """
        accepted_by = ['annotation', 'window', 'bin']
        accepted_metrics = set(['sum', 'std', 'mean', 'median', 'count', 'min', 'max', 'I', 'L', 'dens', 'R', 'RI', 'CPM', 'RPKM'])
        if metric not in accepted_metrics or by not in accepted_by:
            print(metric, by)
            sys.exit('Arguments not recognized, please provide a <metric> included in\n  -'+'\n  -'.join(accepted_metrics)+'\nand <by> from\n  -'+'\n  -'.join(accepted_by)+'\n')

        if by=='annotation':

            # Prepare annotation
            if not annotation_dict:
                annotation_dict = {k:v for k, v in self.annotation.items()}

            # Shorten annotation if required
            if annotation_shortening:
                annotation_dict = self.terminal_shortening(Nterminal=Nterminal, Cterminal=Cterminal, metric=shortening_metric, predict=predict, window_size=window_size, Nmax=Nmax, Cmax=Cmax, inplace=False, plot=False)

            # Add additional annotation
            if additional_annotation:
                self.set_additional_annotation(additional_annotation)
            if self.additional_annotation:
                used_ides = set(annotation_dict.keys())
                for k, v in self.additional_annotation.items():
                    if len(v)!=3 or type(v[0])!=int or type(v[1])!=int or type(v[2])!=str:
                        print('Annotation dictionary format not recognized for entry {}, pass a dictionary {gene:[int(start), int(end), str(strand)]}. Entrt not considered.\n'.format(k))
                    else:
                        if k not in used_ides:
                            annotation_dict[k]=v
                        else:
                            print('Additional annotation entry {} name is already used in main annotation.\n'.format(k))

            # Last step
            results=ignore_metric_for_annotations(self.zreads, annotation_dict, metric=metric, ignore=self.ignore, total_reads=sum(self.reads), gene_len_thr= gene_len_thr)
        else:
            if by=='bin':
                self._update_roll_(window_size=window_size, metric=metric, overlapping=False)
                results = self._bin_roll_
            else:
                self._update_roll_(window_size=window_size, metric=metric, overlapping=True)
                results = self._roll_

        # Apply corrections if required:
        return results

    def metric_by(self, metric='all', by='annotation',
                  normalization='minmax', scaling=False, log2=False, multiplying_factor=1.0,
                  annotation_dict=None, additional_annotation=None,
                  annotation_shortening=True, shortening_metric='RI', Nterminal=5, Cterminal=5,
                  predict=False,
                  bins=False, window_size=50, Nmax=10, Cmax=10,
                  gene_len_thr=12,
                  inplace=True):

        """
        allowed normalization: zscore, minmax, total (default minmax)
        scaling: zscore, minmax, useful to compare between samples
        """

        # Check customs
        _scal_methods = {'zscore':zscore, 'minmax':minmax} # All methods work passing a distribution
        if scaling and scaling not in _scal_methods.keys():
            scaling = 'zscore'
            print('{} not included in accepted methods, using zscore.')
        if metric=='all':
            metric = ['sum', 'std', 'mean', 'median', 'count', 'L', 'min', 'max', 'I', 'dens', 'R', 'RI', 'CPM', 'RPKM']

        # Run
        self.metric         = False
        self.metric_lengths = False
        self.metric_ides    = False
        self.metric_values  = False

        self.metric_lengths = self._single_metric_by('L', by, annotation_dict, additional_annotation, annotation_shortening, shortening_metric, Nterminal, Cterminal, predict, bins, window_size, Nmax, Cmax, inplace=False,  gene_len_thr= gene_len_thr)
        if by=='annotation':
            self.metric_ides = list(self.metric_lengths.keys())
            self.metric_ides.sort(key=lambda x: '{0:0>8}'.format(x).lower())
            self.metric_lengths = np.array([self.metric_lengths[k] for k in self.metric_ides])
        else:
            n = len(self.metric_lengths)
            expected_pad = len(str(n))
            prefix = by[:3]
            self.metric_ides = [prefix+str(i).zfill(expected_pad) for i in range(1, n+1)]

        if type(metric)==str:
            # Single metric
            pre_metric = self._single_metric_by(metric, by, annotation_dict, additional_annotation, annotation_shortening, shortening_metric, Nterminal, Cterminal, predict, bins, window_size, Nmax, Cmax, inplace,  gene_len_thr= gene_len_thr)
            if by=='annotation':
                self.metric_values = {'ides':self.metric_ides, metric:np.array([pre_metric[k] for k in self.metric_ides])}
            else:
                self.metric_values = {'ides':self.metric_ides, metric:np.array(pre_metric)}
        elif hasattr(type(metric), '__iter__'):
            # Multi metric
            self.metric_values = {'ides':self.metric_ides}
            for _m in metric:
                pre_metric = self._single_metric_by(_m, by, annotation_dict, additional_annotation, annotation_shortening, shortening_metric, Nterminal, Cterminal, predict, bins, window_size, Nmax, Cmax, inplace=False,  gene_len_thr= gene_len_thr)
                if by=='annotation':
                    pre_metric = np.array([pre_metric[k] for k in self.metric_ides])
                # Scale if required
                if scaling:
                    if scaling not in _scal_methods.keys():
                        print('{} not included in accepted methods, using zscore.')
                    else:
                        pre_metric = _scal_methods[scaling](pre_metric)
                self.metric_values[_m] = pre_metric
        else:
            sys.exit('Metric format not recognized, pass a string or a sequence like variable (list, tuple, set)')

        self._mby = by

        # Apply corrections if required; this adds norm_metric or GC_metric to the dictionary. Last column will be used as default
        if normalization:
            _metrics = list(self.metric_values.keys())
            for k in _metrics:
                if k not in ['ides']:
                    if normalization in _scal_methods:
                        norm_metric = _scal_methods[normalization](self.metric_values[k])
                    else:
                        norm_metric = self.metric_values[k]/self._single_metric_by(metric=k, annotation_dict={'genome':[1, self.genome_length, '+']}, annotation_shortening=False, inplace=False,  gene_len_thr= gene_len_thr)['genome']
                    
                    if scaling:
                        norm_metric = _scal_methods[scaling](norm_metric)
                    self.metric_values['norm_{}'.format(k)] = norm_metric

        # Last step
        if inplace:
            self._metric_df(self.metric_values, 1, log2=log2)
        else:
            return self._metric_df(self.metric_values, 0, log2=log2)

    def intergenic_annotation(self, annotation_dict=None, prefix='ig', side_removal=150, min_size=100, inplace=True):
        """ Allow to extract intergenic based annotation """
        if annotation_dict:
            coverage_annotation = set(positions_covered_by_annotation(annotation_dict))
        else:
            coverage_annotation = set(positions_covered_by_annotation(self.annotation))
        coverage_annotation = list(set(range(1, self.genome_length+1)).difference(coverage_annotation))
        coverage_annotation = list2ranges(coverage_annotation, annotate=prefix, side_removal=side_removal, min_size=min_size)
        if inplace:
            self.intergenic = coverage_annotation
        return coverage_annotation


    def NRL_for_annotation(self, annotation=None, inplace=False):
        """
        Return the number of positions covered by insertions in annotation (N),
        sum of reads (R) and length of annotation (L).
        annotation ([st, en, strand]) or a dictionary of them.
        Return a dictionary if input dictionary of genes or [N, R, L] if annotation.
        """

        if not annotation:
            annotation = self.annotation
        dictionary = {}
        if type(annotation)==dict:
            for k, v in annotation.items():
                dictionary[k] = self.NRL_for_annotation(v)
            if inplace:
                self.NRL = dictionary
            else:
                return dictionary
        elif type(annotation)==list and type(annotation[0])==int and type(annotation[1])==int and type(annotation[2])==str:
            indices = np.array(positions_covered_by_annotation(annotation))-1       # -1 require to set it to 0 base
            covered = np.take(self.zreads, indices)                                 # Be aware we use the reads with 0 values, required for next step
            L       = len(covered)
            N       = len(covered[covered>1])
            R       = sum(covered)
            return [N, R, L]
        else:
            sys.exit('Wrongly formated annotation, pass dictionary or list')



    # PLOTTING FUNCTIONS
    def plot_read_density(self, bins=None, color='darkcyan', title=None, normed=False, zeroes=False, save=None):
        """ Plot histogram with the read densities """
        if not zeroes:
            _x = self.reads
        else:
            _x = self.zreads

        if bins:
            plthist(_x, bins=bins, color=color, density=normed)
        else:
            plt.hist(_x, color=color, density=normed)
        plt.xlabel('Number Total Reads')

        if normed:
            plt.ylabel('Frequency [%]')
        else:
            plt.ylabel('Number of cases')

        if title:
            plt.title(title)
        else:
            plt.title(self.identifier+' read density histogram')

        if save:
            if save[-1]=='/':
                if title:
                    save += title+'_read_density.svg'
                else:
                    save += self.identifier+'_read_density.svg'
            plt.savefig(save)
        else:
            plt.show()


    def plot_profile(self, gene_id=None, start=None, end=None, log2=True, color='darkcyan', save=False, ylim=False,
                     show_insertions=True,
                     smooth=True, metric='RI', bins=False, color2='blueviolet', extend='auto',
                     Nterminal=5, Cterminal=5, predict=False, window_size=51, Nmax=10, Cmax=10, show_short=None, 
                     show_domains=False, domain_levels='auto', domain_colors='auto', domain_evalue=0.1, show_plot=True,
                     previous_ax=None, _mark_positions=False, __close__=True):
        """
        Custom gene profile plotter.

        <gene_id> to plot that gene profile (it has to be included in annotation). Alternatively, you can provide
        <start> and <end> (base 1) to plot from <start> to <end>.

        <log2> to perform the log2 of reads. Default=True.
        <color> defines the plot color, darkcyan as default, <color2> is for the smooth profile, blueviolet as default
        <save> write the path and extension (pdf, svg, png...) to save the figure to that path
        <ylim> defines the y section we see, it has to be passed as vector

        <show_short> to plot the N and C termninal removal. Default if gene_id is True, false for start-end option (it can be activated).
        <smooth> use <window_size> to plot a smooth profile of the insertions.
        <extend> pb to extend up and downstream the region. If int--> number of bases, if float--> percentage of size if not set, auto. 

        This function also accepts all the commands included in terminal_shortening, instead of
        shortening the annotation, it will plot the limits/percentages that should not be included.
        Check manual for reference about this arguments:
            Nterminal=5, Cterminal=5, predict=False, window_size=50, Nmax=10, Cmax=10

        <show_domains> and <levels> defines the plotting of domains
        """

        # Define basics for plotting
        _whole = False # For ploting the whole genome
        if not gene_id and not (start and end):
            st = 1
            en = int(self.genome_length)
            title = str(self.genome)+' [ '+str(st)+', '+str(en)+' ]'
            predict = False
            _whole = True
            extend = 0
            if smooth:
                show_insertions=False
        elif gene_id:
            if show_short!=False:
                show_short = True     # By default will print alternatives if gene
            if gene_id not in self.annotation:
                sys.exit('Gene identifier not found in annotation, are you sure is properly written? Have you tried using the start and end coordinates?')
            else:
                st, en, strand = self.annotation[gene_id]
                st, en = sorted([int(i) for i in [st, en]])
                title = gene_id+' [ '+str(st)+', '+str(en)+', '+strand+' ]'
        else:
            if not 0<=start<=end<=self.genome_length:
                sys.exit('Coordinates do not match genome length, both should be smaller than '+str(len(self.genome))+'\n')
            else:
                st = start
                en = end
                title = str(self.genome)+' [ '+str(start)+', '+str(end)+' ]'
        l = float(en-st+1)
        if type(extend)==float:
            extend = int(l*extend/100.0)
        elif type(extend)==str:
            extend = int(l*0.05)
        else:
            pass
        if not previous_ax:
            title = self.identifier+'\n'+title

        # Plot
        if not previous_ax:
            f, ax = plt.subplots(1, figsize=(18,3))
        else:
            f, ax = previous_ax
        x = range(st-extend, en+1+extend)
        xlabel = 'genome position [bp]'
        if smooth:
            self._update_roll_(window_size, metric=metric, overlapping=True)
            # print(st-extend-1, en+extend)
            y2 = self._roll_[st-extend-1:en+extend]
            y2label = metric
            if log2:
                y2 = np.log2(y2+1)
                if previous_ax:
                    y2label = self.identifier
                else:
                    y2label = 'log2('+metric+')'

            # print(len(y2), len(x))
            ax.plot(x, y2, color=color2, label=y2label)

        # Alternative start (by percentage), return position + percentages
        if show_short:
            if predict:
                # Nterminal, Cterminal = predict_N_C_terminal(annotation=[st, en, strand], data=self._roll_, window_size=window_size, Nmax=Nmax, Cmax=Cmax)
                Nterminal, Cterminal = predict_N_C_terminal(annotation=[st, en, strand], data=self._roll_, window_size=window_size, Nmax=Nmax, Cmax=Cmax)
            alt_st, alt_en, _ = process_annotation(annotation=[st, en, strand], Nterminal=Nterminal, Cterminal=Cterminal, predict=False)

        if _whole:
            self._update_roll_(1000, metric=metric, overlapping=False) # Bin size of 1000
            y1 = self._bin_roll_
            x  = range(1, len(y1)+1)
            xlabel = 'genome position [bin size='+str(window_size)+']'
            width = 1.8 # matplotlib default
        else:
            y1 = self.zreads[st-extend-1:en+extend]
            width = l/500.0 # custom to avoid artifacts in the representation

        if log2:
            y1 = np.log2(y1+1)
            ylabel = 'log2(reads)'
        else:
            ylabel = 'reads' #TODO new metric here

        plt.ylabel(ylabel, fontsize=18)
        if show_insertions:
            xy1 = np.column_stack((x, y1))
            xy1 = xy1[xy1[:,1]>0.0]
            if len(xy1)>0:
                ax.bar(xy1[:,0], xy1[:,1], color=color, edgecolor="none", label='Insertion')

        # Plot annotation and percentages
        if gene_id:
            cst = 'gray'
            cen = 'k'
            linewidth = 3
            if strand=='+':
                ax.axvline(st, c=cst, label='start', linewidth=linewidth)
                if show_short:
                    ax.axvline(alt_st, c=cst, linestyle='--', label="N'="+str(round(Nterminal, 2))+'%', alpha=0.8, linewidth=linewidth)
                ax.axvline(en, c=cen, label='stop', linewidth=linewidth)
                if show_short:
                    ax.axvline(alt_en, c=cen, linestyle='--', label="C'="+str(round(Cterminal, 2))+'%', alpha=0.8, linewidth=linewidth)
            else:
                ax.axvline(en, c=cst, label='start', linewidth=linewidth)
                if show_short:
                    ax.axvline(alt_en, c=cst, linestyle='--', label="N'="+str(round(Nterminal, 2))+'%', alpha=0.8, linewidth=linewidth)
                ax.axvline(st, c=cen, label='stop', linewidth=linewidth)
                if show_short:
                    ax.axvline(alt_st, c=cen, linestyle='--', label="C'="+str(round(Cterminal, 2))+'%', alpha=0.8, linewidth=linewidth)

        # Ultimate fit and save
        ax.set_xlabel(xlabel, fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(title, fontsize=18)
        ax.set_xlim(min(x), max(x))
        if not previous_ax:
            if __close__==818:
                pass
            else:
                ax.legend(fontsize=18, loc='center left', bbox_to_anchor=(1, 0.5))

        # additional lines
        try:
            for i in _mark_positions:
                ax.axvline(i, c='darkslateblue', alpha=0.8, linewidth=3)
        except:
            pass

        # Plot domain
        if (show_domains and self.domains and gene_id) and gene_id in self.domains:
            your_domains = self.domains[gene_id]
            # define colors and levels
            domnames = list(set([i[0] for i in your_domains]))
            if domain_colors=='auto':
                cmap = cm.get_cmap('mako', len(domnames))
                domnames = {domnames[i]:rgb2hex(cmap(i)[:3]) for i in range(cmap.N)}
            else:
                domnames = {domnames[i]:'darkslateblue' for i in range(len(domnames))}
            # print(domnames)
            if domain_levels=='auto':
                domain_levels = 5

            # Define height and position of the domain box
            rect_y = max(y1)
            if smooth:
                rect_y = max(rect_y, max(y2))
            h = rect_y/10.0
            rect_y = rect_y/-2.0
            # Plot domain
            c = 0   # Factor to avoid overlap
            for domide, domst, domen, domeval in your_domains:
                if type(domain_evalue)==float or type(domain_evalue)==int:
                    if domeval<=domain_evalue:
                        domlen = domen-domst+1
                        # (xy), size of the rect, heigth of the rect
                        rect = plt.Rectangle((domst, rect_y-(h*c*1.5)), domlen, h ,facecolor=domnames[domide], clip_on=False, linewidth=0, alpha=0.8) 
                        ax.add_patch(rect)
                        ax.text(domen+5, rect_y-(h*c*1.5), domide, fontsize=18, color='k')   # Name of the domain at the end of the box slightly displaced
                        c+=1
                        if c==domain_levels:
                            c=0
                else:
                    if domain_evalue[0]<=domeval<=domain_evalue[1]:
                        domlen = domen-domst+1
                        # (xy), size of the rect, heigth of the rect
                        rect = plt.Rectangle((domst, rect_y-(h*c*1.5)), domlen, h ,facecolor=domnames[domide], clip_on=False, linewidth=0, alpha=0.8) 
                        ax.add_patch(rect)
                        ax.text(domen+5, rect_y-(h*c*1.5), domide, fontsize=18, color='k')   # Name of the domain at the end of the box slightly displaced
                        c+=1
                        if c==domain_levels:
                            c=0
        elif show_domains:
            print('No domains plotted, ensure you set the domains (s.set_domains(inFile)) from hmmscan and gene_id is included in that file')

        if save:
            plt.savefig(save, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            if __close__ and __close__!=818:
                plt.close()
            else:
                pass
        return [f, ax]

    def correlate(self, other_dataset, limit=False, color='darkcyan', color2='blueviolet', save=False, show_plot=True):
        """ Correlate position-reads values """
        slope, intercept, r_value, p_value, std_err, x_0, x_1, y_0, y_1 = linear_stats(self.zreads, other_dataset.zreads)
        plt.scatter(self.zreads, other_dataset.zreads, color=color)
        plt.plot([x_0, x_1], [y_0, y_1], color=color2, label='R2 = {:.3f}'.format(r_value**2), linestyle='--')
        plt.xlabel('{} {}'.format(self.identifier, 'reads'))
        plt.ylabel('{} {}'.format(other_dataset.identifier, 'reads'))
        plt.legend()
        if limit:
            if type(limit) in [float, int]:
                plt.ylim(0, limit)
                plt.xlim(0, limit)
            else:
                plt.xlim(0, limit[0])
                plt.xlim(0, limit[1])
        if save:
            plt.savefig(save, bbox_inches='tight')
        if show_plot:
            plt.show()
        return {'slope':slope, 'intercept':intercept, 'r_value':r_value, 'p_value':p_value, 'std_err':std_err}


    def correlate_metric(self, other_dataset, by='annotation', metric='R', other_metric=False, color='darkcyan', color2='blueviolet', save=False, show_plot=True):
        """ Correlate by metric """

        if metric and not other_metric:
            ma = metric
            mb = metric
        elif metric and other_metric:
            ma = metric
            mb = other_metric
        else:
            ma=list(self.metric_values.keys())[1]
            mb=list(other_dataset.metric_values.keys())[1]

        if not self.metric_values or ma not in self.metric_values:
            self.metric_by(by=by, metric=ma, inplace=True)
        if not other_dataset.metric_values or ma not in other_dataset.metric_values:
            other_dataset.metric_by(by=by, metric=mb, inplace=True)

        x = self.metric_values[ma]
        y = other_dataset.metric_values[mb]

        slope, intercept, r_value, p_value, std_err, x_0, x_1, y_0, y_1 = linear_stats(x, y)

        if show_plot or save:
            plt.scatter(x, y, color=color)
            plt.plot([x_0, x_1], [y_0, y_1], color=color2, label='R2 = {:.3f}'.format(r_value**2), linestyle='--')
            plt.xlabel('{} {} by {}'.format(self.identifier, ma, self._mby))
            plt.ylabel('{} {} by {}'.format(other_dataset.identifier, mb, other_dataset._mby))
            plt.legend()

            if save:
                plt.savefig(save, bbox_inches='tight')
            if show_plot:
                plt.show()

        return {'slope':slope, 'intercept':intercept, 'r_value':r_value, 'p_value':p_value, 'std_err':std_err}


    def difference(self, other_dataset, inplace=False, aub=False):
        """ If aub --> keys found in other_dataset are also conserved """
        if aub:
            keys = set(self.profile.keys()).union(other_dataset.profile.keys())
        else:
            keys = self.profile.keys()

        d3 = {key: self.profile.get(key, 0) - other_dataset.profile.get(key, 0) for key in keys}
        if inplace:
            self._update_stats_(d3)
        else:
            new_dataset = self.copy('swallow')
            new_dataset._update_stats_(d3)
            return new_dataset
        
    def addition(self, other_dataset, inplace=False, aub=False):
        if aub:
            keys = set(self.profile.keys()).union(other_dataset.profile.keys())
        else:
            keys = self.profile.keys()
        d3 = {key: self.profile.get(key, 0) + other_dataset.profile.get(key, 0) for key in keys}
        if inplace:
            self._update_stats_(d3)
        else:
            new_dataset = self.copy('swallow')
            new_dataset._update_stats_(d3)
            return new_dataset
        
    # OTHER FUNCTIONS

    def annotation_per_base(self, annotation_dict=None, start=False, end=False, regenerate=False):
        if not self._one_base_annotation_ or regenerate:
            self._one_base_annotation_ = {i:[] for i in range(1, self.genome_length+1)}
            if type(annotation_dict)==dict:
                iterann = annotation_dict.items()
            else:
                iterann = self.annotation.items()
            for k, v in iterann:
                for i in range(v[0], v[1]+1):
                    self._one_base_annotation_[i].append(k)

            if self.ignore:
                for i in self.ignore:
                    self._one_base_annotation_[i]=[]

            if start and end:
                return {k:v for k, v in self._one_base_annotation_.items() if start<=k<=end}
            else:
                return self._one_base_annotation_
        else:
            if start and end:
                return {k:v for k, v in self._one_base_annotation_.items() if start<=k<=end}
            else:
                return self._one_base_annotation_

    def prioritize_regions_by_metric(self, metric='RI', window_size=51, percentile=50, region_type='both', size=1000, add_annotations=True):
        """
        Basic function to prioritize whole genome NE and E regions.
        Arguments include:
        metric = sum, std, mean, median, count, min, max, I, R, RI
            window_size for overlapping windows
            percentile = threshold to select the threshold to filter metrics
            region_type =  NE, E or both
            size = filter by size the candidate regions
        """
        if type(region_type)==str:
            self._update_roll_(window_size, metric=metric, overlapping=True)
            threshold = np.percentile(self._roll_, percentile)

            if region_type not in ['NE', 'E', 'both']:
                sys.exit('Provide a region type, either NE or E')
            elif region_type!='both':
                if region_type=='NE':
                    ranges = [i for i in range(self.genome_length) if self._roll_[i]>=threshold]
                else:
                    ranges = [i for i in range(self.genome_length) if self._roll_[i]<=threshold]
                processed = {}
                c=1

                # Lambda works different, first line for python 2
                for k, g in groupby(enumerate(ranges), lambda i_x:i_x[0]-i_x[1]):
                    label = 'region_'+region_type+'_'+str(c)
                    f = list(map(itemgetter(1), g))
                    mi = min(f)
                    ma = max(f)
                    ll = ma-mi+1
                    if ll>=size:
                        processed[label] = [label, mi, ma, ll, region_type]
                        if add_annotations:
                            affected_anns = lists2list(self.annotation_per_base(start=mi, end=ma).values())
                            if affected_anns:
                                affected_anns = '/'.join(sorted(map(str, affected_anns)))
                                processed[label].append(affected_anns)
                            else:
                                processed[label].append('')
                        c+=1
                df = pd.DataFrame.from_dict(processed, orient='index')
                if add_annotations:
                    df.columns = ['region', 'start', 'end', 'size', 'type', 'annotations']
                else:
                    df.columns = ['region', 'start', 'end', 'size', 'type']
                df.sort_values(by=['size'], ascending=False, inplace=True)
                return df
            else:
                processedE  = self.prioritize_regions_by_metric(metric=metric, window_size=window_size, percentile=percentile, region_type='E', size=size)
                processedNE = self.prioritize_regions_by_metric(metric=metric, window_size=window_size, percentile=percentile, region_type='NE', size=size)
                df = pd.concat([processedE, processedNE])
            return df
        elif type(region_type)==list:
            lol = []
            for i in [metric, window_size, percentile, region_type, size]:
                if type(i)!=list:
                    lol.append([i])
                else:
                    lol.append(i)
            lol = prepare_lol(lol)
            dfs = []
            for metric, window_size, percentile, region_type, size in zip(*lol):
                dfs.append(self.prioritize_regions_by_metric(metric=metric, window_size=window_size, percentile=percentile, region_type=region_type, size=size))
            df = pd.concat(dfs)
            df.drop_duplicates(inplace=True)
            df.sort_values(by=['start'], inplace=True)
            return df
        else:
            pass

    def write_profile(self, path):
        with open(path, 'w') as fo:
            for k, v in self.profile.items():
                fo.write('{}\t{}\n'.format(k, v))

    def save_dataset(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


#####
# RAUC block
#####

    def range_RAUC(self, start, end, strand, thresholds=False, metric='I', frame=True,
                   max_value_reads=False, by_percentile=False, step=5, __auc__=1):
        # Define span
        gene_reads = self.zreads[start-1:end]
        if strand=='-' or strand==0:
            gene_reads = gene_reads[::-1]
        # Define threshold if required
        if not thresholds:
            thresholds = define_thresholds(gene_reads, max_value_reads, by_percentile, step)

        # Calculate the metric
        if __auc__:
            return np.around(np.trapz(read_decay(gene_reads, thresholds, metric=metric, frame=frame, total_reads=sum(self.reads)), x=thresholds), decimals=2)
        else:
            return np.around(read_decay(gene_reads, thresholds, metric=metric, frame=frame, total_reads=sum(self.reads)), decimals=2)

    def RAUC(self, start=None, end=None, strand=None, gene_id=None, annotation=None,
            level='gene',
            frame=True,
            thresholds=False,
            metric='I',
            max_value_reads=False, by_percentile=False, step=5
            ):
        """
        level = [sample, gene]
        """
        # define thresholds
        flag = False
        if type(annotation)==dict:
            pass
        elif type(start)==int and type(end)==int and type(strand)==str:
            annotation = {'_':[start, end, strand]}
            flag = '_'
        elif type(gene_id)==str:
            annotation = {gene_id:self.annotation[gene_id]}
            flag = gene_id
        else:
            annotation = self.annotation

        # Define general thresholds if by sample
        if level=='sample':
            # Same thresholds for all
            if not thresholds:
                thresholds = define_thresholds(self.zreads, max_value_reads, by_percentile, step)

        # Run the process
        rs = {k:self.range_RAUC(v[0], v[1], v[2], thresholds, metric, frame, max_value_reads, by_percentile, step) for k, v in annotation.items()}
        if flag:
            return rs[flag]
        else:
            return rs

    def frame_metric(self, start=None, end=None, strand=None, gene_id=None, annotation=None, metric='I'):
        # define thresholds
        flag = False
        if type(annotation)==dict:
            pass
        elif type(start)==int and type(end)==int and type(strand)==str:
            annotation = {'_':[start, end, strand]}
            flag = '_'
        elif type(gene_id)==str:
            annotation = {gene_id:self.annotation[gene_id]}
            flag = gene_id
        else:
            annotation = self.annotation

        # Run the process
        rs = {k:self.range_RAUC(v[0], v[1], v[2], thresholds=0.1, metric=metric, frame=True, __auc__=0) for k, v in annotation.items()}
        if flag:
            return rs[flag]
        else:
            return rs

### RAUC functions

def define_thresholds(distribution, max_value=False, by_percentile=False, step=1):

    if by_percentile:
        thresholds = [i[1] for i in percentile_iterator(distribution, step=step)]
    elif max_value:
        thresholds = range(0, max_value, step)
    else:
        thresholds = range(0, int(max(distribution)), step)
    return thresholds

def percentile_iterator(values, step=5):
    """
    Return percentiles from 0 to 100 with <step> for specific values
    for <values> distribution.

    Step = 5 means that it returns percentiles 0, 5, 10, ..., 90, 95, 100
    """
    for i in range(0, 100+step, step):
        yield i, np.percentile(values, i)

def read_decay(your_reads, thresholds, metric='I', frame=False, total_reads=0, ignore=None,  gene_len_thr=10):
    """ Return metric for section of reads passing threshold """
    if not frame:
        if thresholds==0.1:
            return ignore_metric_for_annotations(your_reads, {'_':[1, len(your_reads), '+']}, metric, ignore=ignore, total_reads=total_reads,  gene_len_thr= gene_len_thr)['_']
        else:
            return np.array([ignore_metric_for_annotations(your_reads[your_reads>thr], {'_':[1, None, '+']}, metric, ignore=ignore, total_reads=total_reads)['_'] for thr in thresholds]) # None as end of the annotation ensures it takes the whole read array
    else:
        return [read_decay(your_reads[a::b], thresholds, metric, total_reads=total_reads) for a, b in zip([0,0,1,2], [None,3,3,3])]

######
# RAUC block ends
#####



class collection(object):
    """
    Collection of datasets
    You can pass a template collection file through inFile or pass
    many lists with same length to form the dataset. Location of the datasets
    can be passed through the inFile argument as well.
    inFile and genome location are mandatory.
    If a specific list is not provided, fill it with 0s.
    If instead of a list, a single value is passed, it will be considered the same
    for all the samples.
    """
    # Attributes
    def __init__(self         , location, genome=[] , log=[]     , goldset=[]  , annotation=[]   ,
                 identifier=[], time=[] , passage=[], dilution=[], replicate=[], doubling_time=[],
                 condition=[] , treatment=[]):


        # Define the attributes
        dtypes = {'identifier':str, 'location':str , 'log':str       , 'genome':str     , 'annotation':str     , 'goldset':str,
                  'time':float    , 'passage':float, 'dilution':float, 'replicate':float, 'doubling_time':float,
                  'condition':int , 'treatment':str}   # EDIT HERE IF NEW FIELDS ADDED
        if type(location)==str:
            df, lol, goldset = load_collection_from_file(location, dtypes=dtypes)    # If provided in second sheet, goldset will be uploaded here
        elif type(location)==list:

            # We need lists:

            # location.sort()
            if len(identifier)==0:
                identifier = [i.split('/')[-1].split('.')[0].split('read')[0] for i in location]
            if len(genome)==0:
                sys.exit('Provide genome(s)')

            your_locals = locals()
            your_locals = {k:[v] if type(v)==str else v for k, v in your_locals.items()} # To avoid error if a string, in that case just trail that str to all
            lol = [your_locals[i] for i in ['identifier', 'location', 'log', 'genome', 'annotation',
                                            'goldset', 'time', 'passage', 'dilution', 'replicate',
                                            'doubling_time', 'condition', 'treatment']]
            lol = prepare_lol(lol)
            df  = pd.DataFrame(lol).transpose()
            df.columns = ['identifier', 'location' , 'log'     , 'genome'   , 'annotation'   , 'goldset',
                          'time'      , 'passage'  , 'dilution', 'replicate', 'doubling_time',
                          'condition' , 'treatment']   # EDIT HERE IF NEW FIELDS ADDED
            df = df.set_index('identifier', drop=False)
        else:
            sys.exit('Provide a template file path or a list of paths')

        # Define some attributes
        self.dataframe = df

        # Load the single datasets:
        print('Loading datasets...')
        datasets  = {}
        resources = self.__faster_dataset_load__(lol[3], lol[4], lol[5], location)
        for list_of_attrs in zip(*lol):
            # * required to interpret the list of lists as single lists
            datasets[list_of_attrs[0]]= dataset(location=list_of_attrs[1]  , log=list_of_attrs[2]       , genome=list_of_attrs[3]        , genome_sequence=resources['geno'][list_of_attrs[3]],
                                                annotation=resources['anno'][list_of_attrs[4]]          , time=list_of_attrs[6]          , passage=list_of_attrs[7],
                                                dilution=list_of_attrs[8]  , replicate=list_of_attrs[9] , doubling_time=list_of_attrs[10],
                                                identifier=list_of_attrs[0], condition=list_of_attrs[11], treatment=list_of_attrs[12]    ,
                                                goldset=resources['gold'][list_of_attrs[5]])
        print('All datasets loaded.')
        self.datasets = datasets

        # Colormap consistent along datasets
        ides = list(self.datasets.keys())
        cmap = cm.get_cmap('winter', len(self.datasets))
        self.cmap = {ides[i]:rgb2hex(cmap(i)[:3]) for i in range(cmap.N)}


    # OTHER FUNCTIONS
    def __faster_dataset_load__(self, genome_paths, annotation_paths, goldset_paths, inFile):
        """
        Interpret the list of lists to only load to memory one genome/annotation dict as usual
        all the genomes in a collection will be the same.
        The list of lists is exprected to have paths to the elements in standardized format.
        Returns a dictionary: {'path':element} that can be passed directly later
        Preference for annotation is: genbank < annotation file
        Preference for goldset is: sheet2 < goldset file
        """
        # Reduce at minimum
        annotation_paths = set(annotation_paths)
        genome_paths     = set(genome_paths)
        goldset_paths    = set(goldset_paths)

        # Load the expected elements in each
        path_to_resource = {'anno':{}, 'geno':{}, 'gold':{}}

        for geno in genome_paths:
            path_to_resource['geno'][geno] = load_genome_sequence(geno)
        for anno in annotation_paths:
            if anno.endswith('gb') or anno.endswith('gbk') or anno.endswith('genbank'):
                path_to_resource['anno'][anno] = load_annotation_from_genbank(anno)
            else:
                path_to_resource['anno'][anno] = load_annotation_from_file(anno)
        for gold in goldset_paths:
            if type(gold) is str and os.path.isfile(gold) and gold!='0':
                path_to_resource['gold'][gold] = load_goldset_from_file(gold)
            else:
                # Take it from the excel if possible
                if type(inFile)!=list and (inFile.endswith('.xlsx') or inFile.endswith('.xls')):
                    path_to_resource['gold'][gold] = load_goldset_from_file(inFile)
                else:
                    path_to_resource['gold'][gold] = {}
        return path_to_resource



    def plot_profile(self, gene_id=None, start=None, end=None, log2=True, save=False, ylim=False, cmap=False,
                     show_insertions=True, smooth=True, metric='RI', bins=False, extend='auto',
                     Nterminal=5, Cterminal=5, predict=False, window_size=51, Nmax=10, Cmax=10, show_short=None,
                     show_domains=False, domain_levels='auto', domain_evalue=0.1):
        """ Plot multiple samples together """
        f, ax = plt.subplots(figsize=(18,3))
        first_done = False
        if not cmap or len(cmap)<=len(self.cmap):
            cmap = self.cmap
        for ide, dat in self.datasets.items():
            col = cmap[ide]
            if not first_done:
                ax = dat.plot_profile(gene_id=gene_id, start=start, end=end, log2=log2, color=col,
                                      save=False, ylim=ylim, show_insertions=show_insertions, smooth=smooth,
                                      metric=metric, bins=bins, color2=col, extend=extend,
                                      Nterminal=Nterminal, Cterminal=Cterminal, predict=predict,
                                      window_size=window_size, Nmax=Nmax, Cmax=Cmax, show_short=show_short,
                                      show_domains=show_domains, domain_levels=domain_levels, domain_evalue=domain_evalue,
                                      show_plot=False, previous_ax=ax)
                first_done=True
            else:
                ax = dat.plot_profile(gene_id=gene_id, start=start, end=end, log2=log2, color=col,
                                      save=False, ylim=ylim, show_insertions=show_insertions, smooth=smooth,
                                      metric=metric, bins=bins, color2=col, extend=extend,
                                      Nterminal=False, Cterminal=False, predict=False,
                                      window_size=window_size, Nmax=False, Cmax=False, show_short=False,
                                      show_plot=False, previous_ax=ax)
        if save:
            plt.savefig(save, bbox_inches='tight')
        plt.show()

    # WRITERS



    # TODO Make a function to report dataframe with coverages, values of reads, etx...



####
# ANALYSIS METHOD
####

# Poisson

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)


def poisson_acc_probability(N, L, r):
    mu = r*L
    return poisson.cdf(N, mu)

def poisson_probability(N, L, r):
    mu = r*L
    return poisson.pmf(N, mu)

def return_poisson_class(N, L, rE, rNE, thr=0.0):
    rg = 1.0*N/L
    pe  = poisson_probability(N, L, rE)
    pne = poisson_probability(N, L, rNE)
    # Assign class:
    if rg<=rE:
        clase = 'E'
    elif rg>=rNE:
        clase = 'NE'
    else:
        if pe>thr and pne==0:
            clase = 'E'
        elif pe==0 and pne>thr:
            clase = 'NE'
        elif pe==0 and pne==0:
            clase = 'F'
        else:
            clase = 'F'
    a = [round(rg, 5), pe, pne, clase]
    return a


def poisson_essentiality(your_dataset, mode='GDP', **kwargs):
    """
    Different ways:
        Gene dependent:
            - E and NE as training set of genes (Gene-Dependent Poisson, GDP)
        Gene independent (Gene-Independent Poisson):
            - Extract metric from bimodal distribution model (Bimodal Gene-Independent Poisson, BGIP)
            - Assume E density equal to 0 and NE density from intergenic regions (Intergenic-dependent Poisson, IDP)
        Hybrid:
            - E genes and NE density assumed from intergenic regions (Hybrid Essential Genes Poisson, HEGP)
            - NE genes and E density equal to 0 (Hybrid Non-Essential Genes Poisson, HNGP)
    """
    metric = [kwargs.get('metric', 'I'), 'L']
    if not isinstance(your_dataset.metric, pd.DataFrame):
        print('Defining default metric')
        your_dataset.metric_by(metric, by=kwargs.get('by', 'annotation'),
                               annotation_dict=kwargs.get('annotation_dict', None),
                               additional_annotation=kwargs.get('additional_annotation', None),
                               annotation_shortening=kwargs.get('annotation_shortening', True),
                               shortening_metric=kwargs.get('shortening_metric', 'RI'),
                               bins=kwargs.get('bins', False),
                               window_size=kwargs.get('window_size', 50),
                               predict=kwargs.get('predict_shortening', False),
                               Nterminal=kwargs.get('Nterminal', 5), Cterminal=kwargs.get('Cterminal', 5),
                               Nmax=kwargs.get('Nmax', 10), Cmax=kwargs.get('Cmax', 10),
                               inplace=True)

    if mode not in ['GDP', 'BGIP', 'IDP', 'HEGP', 'HNGP']:
        sys.exit('No available mode '+mode+'\n')

    # We can extract intergenic regions and add them to the additional, (_ig) to detect them
    # we should ensure a minimum region size

    Ns = your_dataset.metric_values[kwargs.get('metric', 'I')]
    Ls = your_dataset.metric_values['L']

    if mode=='GDP':
        if len(set(['E', 'NE']).intersection(set(your_dataset.goldset.values())))==2:
            maskE = np.in1d(your_dataset.metric_ides, your_dataset.goldE).astype(bool)
            maskNE = np.in1d(your_dataset.metric_ides, your_dataset.goldNE).astype(bool)
            _re = np.mean(np.array(Ns)[maskE]/np.array(Ls)[maskE])
            _rne = np.mean(np.array(Ns)[maskNE]/np.array(Ls)[maskNE])
        else:
            sys.exit('GDP mode requires a goldset with NE and E genes\n')
    elif mode=='BGIP':
        densities = Ns/Ls
        y, x, _ = plt.hist(densities,100,alpha=.3,label='data')
        y = np.array([y[2], y[1]]+list(y))
        x = np.array([x[0]-x[2], x[0]-x[1]]+list(x))
        d = dict(zip(x, y))
        x = (x[1:]+x[:-1])/2 # for len(x)==len(y)

        E   = densities[densities<0.5]           # Densities of potential e genes
        NE  = densities[densities>0.5]
        mE , sE  = norm.fit(E)
        mNE, sNE = norm.fit(NE)
        pE  = max({k:v for k, v in d.items() if k < mE+sE  }.values())+2
        pNE = max({k:v for k, v in d.items() if mNE-sNE < k}.values())+2

        expected = (mE, sE, pE, mNE, sNE, pNE)

        params,cov=curve_fit(bimodal,x,y, p0=expected)
        sigma=np.sqrt(np.diag(cov))
        plt.plot(x,bimodal(x,*params),color='red',lw=3,label='model')
        plt.title('kernel-based prediction of GS + Poisson')
        plt.legend()

        df  = pd.DataFrame(data={'params':params,'sigma':sigma},index=bimodal.__code__.co_varnames[1:])
        _re  = df.loc['mu1', 'params']
        _rne = df.loc['mu2', 'params']
    elif mode=='IDP':
        pass

    return {ide:return_poisson_class(N, L, _re, _rne) for ide, N, L in zip(your_dataset.metric_ides, Ns, Ls)}



###################


class protocol(object):
    """
    General class for protocol, it works as parent class for each method.
    Arguments required:
        - preprocessing: filters applied to the dataset (gene_filter, read_filter)
        - by: annotation to use (gene, window, bin)
        - metric:
        - normalization
        - method
    """
    # Attributes

    # Datasets and collections have to be processed in the same manner, the output will always be: {'identifier':output}
    # Output can be anything
    def __init__(self, data, **kwargs):
        # Detect proper datatype
        self.original = copy.copy(data)
        if isinstance(data, dataset):
            self.datasets = {self.original.identifier:copy.copy(self.original)}
        elif isinstance(data, collection):
            self.datasets = {k:copy.copy(v) for k, v in data.datasets.items()}
        elif isinstance(data, protocol):
            self.datasets = {k:copy.copy(v) for k, v in data.datasets.items()}
        else:
            sys.exit('Please provide an accepted datatype from the package: dataset, collection.')

        # Exit if not kwargs provided or a term not found in the dictionary of methods
        protocol_dictionary = {'method':{'poisson':poisson_essentiality}}
        # order:
        # 0-read, 1-gene, 2-tail
        protected_arguments = set(list(protocol_dictionary.keys())+
                                  ['filter_by',
                                   'min_value', 'max_value',
                                   'percentile' , 'list_of_genes', 'predict_shortening', 'Nterminal', 'Cterminal',
                                   'Nmax', 'Cmax', 'window_size',
                                   'lpercentile', 'rpercentile',
                                   'metric', 'by', 'normalization', 'scaling', 'multiplying_factor', 'log2',
                                   'annotation_dict', 'additional_annotation', 'annotation_shortening',
                                   'shortening_metric', 'bins',
                                   'method', 'mode',
                                   'evaluate'])
        # Check basic protocol arguments
        if len(kwargs)==0:
            sys.exit('No information found for the protocol, please specify.')
        else:
            for option, argument in kwargs.items():
                if option not in protected_arguments:
                    sys.exit('Error in "'+str(option)+'='+str(argument)+'". Option "'+str(option)+'" not found.')


        # GC correction
        if 'k' in kwargs:
            for ide in self.datasets.keys():
                self.datasets[ide].GC(k=kwargs.get('k', 4), inplace=True)
       # TODO: implement save, show_plot and colors, general for all
       # def GC(self, k=4, color='darkcyan', color2='blueviolet', save=False, show_plot=True, annotation_dict=None, prefix='ig', side_removal=150, min_size=100, inplace=True):


        # Preprocessing/filters
        if 'filter_by' in kwargs:
            for ide in self.datasets.keys():
                self.datasets[ide].filter(filter_by=kwargs.get('filter_by', 'genes'),
                                          min_value=kwargs.get('min_value', 16), max_value=kwargs.get('max_value', False),
                                          lpercentile=kwargs.get('lpercentile', 5), rpercentile=kwargs.get('rpercentile', 95),
                                          percentile=kwargs.get('percentile', 95),
                                          list_of_genes=kwargs.get('list_of_genes'),
                                          auto=kwargs.get('predict_shortening', False),
                                          Nterminal=kwargs.get('Nterminal', 5), Cterminal=kwargs.get('Cterminal', 5),
                                          window_size=kwargs.get('window_size', 50),
                                          Nmax=kwargs.get('Nmax', 10), Cmax=kwargs.get('Cmax', 10),
                                          inplace=True)
        # By gene, window, bin
        if 'by' in kwargs:
            for ide, your_dataset in self.datasets.items():
                your_dataset.metric_by(metric=kwargs.get('metric', 'dens'), by=kwargs.get('by', 'annotation'),
                                       normalization=kwargs.get('normalization', False),
                                       scaling=kwargs.get('scaling', False), multiplying_factor=kwargs.get('multiplying_factor', 1.0),
                                       log2=kwargs.get('log2', False),
                                       annotation_dict=kwargs.get('annotation_dict', None),
                                       additional_annotation=kwargs.get('additional_annotation', None),
                                       annotation_shortening=kwargs.get('annotation_shortening', True),
                                       shortening_metric=kwargs.get('shortening_metric', 'RI'),
                                       bins=kwargs.get('bins', False),
                                       window_size=kwargs.get('window_size', 50),
                                       predict=kwargs.get('predict_shortening', False),
                                       Nterminal=kwargs.get('Nterminal', 5), Cterminal=kwargs.get('Cterminal', 5),
                                       Nmax=kwargs.get('Nmax', 10), Cmax=kwargs.get('Cmax', 10),
                                       inplace=True)


        # Prediction
        self.results = {}
        if 'method' in kwargs:
            for ide, your_dataset in self.datasets.items():
                self.results[ide] = protocol_dictionary['method'][kwargs.get('method')](your_dataset, mode=kwargs.get('mode', 'GDP'))


        # Evaluate
        if 'evaluate' in kwargs:
            pass



#####################
#     EXECUTION     #
#####################

if __name__=='__main__':

    if len(sys.argv)>1:
        if sys.argv[1]=='test':

            # Small test to check the insertions loader
            insertions_loaded = load_ins_file('./tests/test001.qins')
            assert insertions_loaded == {1:18, 2:81, 3:18}

            # Collections loader
            dtypes = {'identifier':str, 'location':str , 'log':str       , 'genome':str     , 'annotation':str     , 'goldset':str,
                      'time':float    , 'passage':float, 'dilution':float, 'replicate':float, 'doubling_time':float,
                      'condition':int , 'treatment':str}   # EDIT HERE IF NEW FIELDS ADDED
            assert load_collection_from_file('./tests/collection.csv',dtypes)[1]==load_collection_from_file('./tests/collection.xlsx', dtypes)[1]
            # [1] is required as dataframes cannot be asserted, we assert over the list of lists


            # Windows loader:
            assert windows([1,2,0,0,2,1], 5, metric='count')==np.array([ 5.,  5.,  5.,  5.,  5.,  5.])
            assert windows([1,2,0,0,2,1], 5, metric='I')==np.array([ 4.,  3.,  3.,  3.,  3.,  4.]) 
