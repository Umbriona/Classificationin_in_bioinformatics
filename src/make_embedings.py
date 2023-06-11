import transformers

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import umap

from sklearn.manifold import TSNE
from transformers import pipeline
from Bio import SeqIO

import argparse

parser = argparse.ArgumentParser( 
    """ This script will read a fasta file """)

parser.add_argument("-i", "--input", type = str, required = True, help = 
                    "Path to fasta file to be converted")
parser.add_argument("-o", "--output", type = str, default = "./embedings.tfrecords", help = 
                   "Path to output ")

NON_STANDARD_AMINO = ["B", "U", "Z", "X"]


def read_fasta(args):
    
    df_fasta = { "id":[], 'seq':[], "TM":[]}
    for rec in SeqIO.parse(input, "fasta"):
        df_fasta["id"].append(rec.id)
        sequence = str(rec.seq)
        
        if NON_STANDARD_AMINO in sequence:
            raise ValueError (f"Sequences can not contain non standard amino acids")
            
        df_fasta["seq"].append(str(rec.seq))
    df_fasta = pd.DataFrame(df_fasta) 
    return df_fasta



def creat_embedings(df_fasta):
    
    ## Set transformer parameters 
    transformers.logging.set_verbosity_error()
    pipeline = pipeline('feature-extraction', model='facebook/esm1b_t33_650M_UR50S')

    embeddings_list = []
    # Exctract CLS embeddings from ESM
    for _, row in df.iterrows():
        print (_)
        prot = row['seq']
        this_embedding = pipeline(prot)
        ## CLS extaction
        cls_token = this_embedding[0][0]
        embeddings_list.append(cls_token)
    assert len(embeddings_list) == len(df)
    df_fasta["Embedding"] = embedding_list
    return df_fasta

def write_records():
    

def main(args):
    
    df_fasta = read_fasta(args)
    df_fasta = creat_embedings(df_fasta)
    
    return 0

if __name__ == "__main__":
    main()















