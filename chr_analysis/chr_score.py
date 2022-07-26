
import numpy as np
import pandas as pd
from data.get_feature import feature_extract
from Bio import SeqIO
from xgboost import XGBClassifier as XGB
import multiprocessing


def pred_chr(name, stride=2000):
    file_name = r'.\chromosome\\' + name
    save_name = r'.\chr_score\\' + name

    with open( file_name+'.fa','r' ) as fa:
        for record in SeqIO.parse(fa, 'fasta'):
            fa = str(record.seq)
    num = int(np.ceil( len(fa)/stride - 0.1))
    seqs = [fa[i*stride:(i+1)*stride] for i in range(num)]
    feature = feature_extract(seqs)
    np.save( save_name+'.npy', feature )

    feature = np.load( save_name+'.npy' )
    idx = np.load(r'feature_idx.npy')
    feature = feature[:,idx]
    model = XGB()
    model.load_model('model.json')
    y_prob = model.predict_proba(feature)[:,1]
    data = pd.DataFrame(y_prob)
    data.to_excel(save_name+'_result.xlsx', header=False, index=False)
    print('finish '+name)


def cut_sum(names):
    bins = [ i*0.1 for i in range(11) ]
    bins[0] = 0.01
    
    all = []
    for name in names:
        file = r'.\chr_score\\' + name + '_result.xlsx'
        df = pd.read_excel( file,header=None,usecols=None,dtype='float')
        scores = df.iloc[:,0].values
        cuts = pd.cut( scores, bins )
        counts = pd.value_counts( cuts, normalize=True, sort=False )
        one = list(dict(counts).values())
        one.insert(0, name)
        all.append(one)

    header = list(dict(counts).keys())
    header.insert(0,'name')

    writer = pd.ExcelWriter(r'./results/chr_results.xlsx')
    data = pd.DataFrame(all)
    data.to_excel(writer, header=header, index=False)
    writer.close()


if __name__ == '__main__':
    names = ['Athaliana_chr2', 'Celegans_chrII','DNA_storage','Drosophila_chrY',\
                'Human_chr22', 'JCVI_syn1.0', 'Oryza_sativa_chr9', \
                'Yeast_synV', 'Yeast_synX', 'Mouse_chr19', 'CETH_2.0']
    stride = 2000
    record = []
    for name in names:
        process = multiprocessing.Process(target=pred_chr, args=(name, stride))
        process.start()
        record.append(process)
    
    for process in record:
        process.join()
    
    cut_sum(names)