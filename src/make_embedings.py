import transformers

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import pickle

from transformers import pipeline
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, AutoModelForMaskedLM

import torch
import os
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel

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
    list_data = []
    for rec in SeqIO.parse(file, "fasta"):
        df_fasta["id"].append(rec.id)
        sequence = str(rec.seq)
        
        if any(amino in sequence for amino in NON_STANDARD_AMINO):
            raise ValueError (f"Sequences can not contain non standard amino acids")
            
        df_fasta["seq"].append(str(rec.seq))
        df_fasta["TM"].append(float(rec.description.split(" ")[-1]))
        list_data.append((rec.id, str(rec.seq)))
    return df_fasta, list_data



def creat_embedings(df_fasta, list_data = None):
    if list_data == None:
        ## Set transformer parameters 
        transformers.logging.set_verbosity_error()

        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
        
        pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, framework="pt",  device=1)

        embeddings_list = []
        # Exctract CLS embeddings from ESM
        for idx, prot in enumerate(df_fasta["seq"]):
            this_embedding = pipe(prot)
            print(prot)
            ## CLS extaction
            #cls_token = this_embedding[0][0]

            #print(this_embedding)
            
            avg_token = np.mean(np.array(this_embedding[0]), axis=0)
            embeddings_list.append(avg_token)

            if idx % 100 ==0:
                print(f"Done with {idx} Embeddings")
        assert len(embeddings_list) == len(df_fasta["seq"])
        df_fasta["Embedding"] = embeddings_list
        return df_fasta
    else:
        print("Using new emb")
        model, alphabet = torch.hub.load("facebookresearch/esm", "esm1_t34_670M_UR50S")
        model = model.to(device = 1)
        batch_converter = alphabet.get_batch_converter()
        results = []
        for idx, data in enumerate(list_data):
            batch_labels, batch_strs, batch_tokens = batch_converter([data])
            with torch.no_grad():
                result = model(batch_tokens.to(device=1), repr_layers=[33], return_contacts=False)
            result = np.mean(result['representations'][33].cpu().numpy(), axis = 1)

            result = [list(emb) for emb in result]
            results.append(result[0])
            if idx % 100 ==0:
                print(f"Done with {idx} Embeddings")
        df_fasta["Embedding"] = results
        return df_fasta


        
def write_pickel(df_fasta, output):
    with open(output, 'wb') as f:
        pickle.dump(df_fasta, f)
    

def main(args):
    
    df_fasta, list_data = read_fasta(args.input)
    #df_fasta = creat_embedings(df_fasta)
    df_fasta = creat_embedings(df_fasta, list_data)
    write_pickel(df_fasta, args.output)
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)






#    else:
#        print("Using new emb")
#        model, alphabet = torch.hub.load("facebookresearch/esm", "esm1_t34_670M_UR50S")
#        batch_converter = alphabet.get_batch_converter()
#        batch_labels, batch_strs, batch_tokens = batch_converter(list_data)
#        with torch.no_grad():
#            results = model(batch_tokens.to(device="cpu"), repr_layers=[33], return_contacts=False)
#        result = np.mean(results['representations'][33].numpy(), axis = 1)
#        result = [list(emb) for emb in result]
#        df_fasta["Embedding"] = result
#        return df_fasta











