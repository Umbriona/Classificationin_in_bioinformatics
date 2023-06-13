import transformers

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import pickle

from transformers import pipeline
from Bio import SeqIO

import argparse

parser = argparse.ArgumentParser( 
    """ This script will read a fasta file """)

parser.add_argument("-i", "--input", type = str, required = True, help = 
                    "Path to fasta file to be converted")
parser.add_argument("-o", "--output", type = str, default = "./embedings.pkl", help = 
                   "Path to output ")

NON_STANDARD_AMINO = ["B", "U", "Z", "X"]


def read_fasta(file):
    
    df_fasta = { "id":[], 'seq':[], "TM":[]}
    for rec in SeqIO.parse(file, "fasta"):
        df_fasta["id"].append(rec.id)
        sequence = str(rec.seq)
        
        if any(amino in sequence for amino in NON_STANDARD_AMINO):
            raise ValueError (f"Sequences can not contain non standard amino acids")
            
        df_fasta["seq"].append(str(rec.seq))
        df_fasta["TM"].append(float(rec.description.split(" ")[-1]))
    return df_fasta



def creat_embedings(df_fasta):
    
    ## Set transformer parameters 
    transformers.logging.set_verbosity_error()
    pipe = pipeline('feature-extraction', model='facebook/esm1b_t33_650M_UR50S', device=1)

    embeddings_list = []
    # Exctract CLS embeddings from ESM
    for idx, prot in enumerate(df_fasta["seq"]):
        this_embedding = pipe(prot)
        ## CLS extaction
        cls_token = this_embedding[0][0]
        embeddings_list.append(cls_token)
        if idx % 100 ==0:
            print(f"Done with {idx} Embeddings")
    assert len(embeddings_list) == len(df_fasta["seq"])
    df_fasta["Embedding"] = embeddings_list
    return df_fasta

def write_pickel(df_fasta, output):
    with open(output, 'wb') as f:
        pickle.dump(df_fasta, f)
    

def main(args):
    
    df_fasta = read_fasta(args.input)
    df_fasta = creat_embedings(df_fasta)
    write_pickel(df_fasta, args.output)
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)















