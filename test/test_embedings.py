import os, sys
path_utils = os.path.dirname(os.getcwd())
sys.path.append(path_utils)
import unittest
import pandas as pd
from src.make_embedings import read_fasta, creat_embedings


# TEST of test1.fasta
TEST_FASTA_1 = "test1.fasta"
TEST_FASTA_2 = "test2.fasta"
TEST_FASTA_1_ROWS = 2
TEST_FASTA_1_COLUMNS = 3
TEST_FASTA_1_SEQUENCE_0 = "SANDRA"
TEST_FASTA_1_ID_1 = "sequence1"
TEST_FASTA_1_TM_1 = 90.0



# Testing Embeding

TEST_FASTA_3 = "test3.fasta"
TEST_FASTA_3_ROWS = 2
TEST_FASTA_3_COLUMNS = 4
TEST_FASTA_3_EMBEDDING_SIZE = 1280


class TestReadFasta(unittest.TestCase):
    def test_shape(self): 
        df_fasta = read_fasta(TEST_FASTA_1)
        self.assertEqual(TEST_FASTA_1_ROWS, len(df_fasta["seq"]))
        self.assertEqual(TEST_FASTA_1_COLUMNS, len(df_fasta.keys()))
        
    def test_read(self):
        df_fasta = read_fasta(TEST_FASTA_1)
        self.assertEqual(TEST_FASTA_1_SEQUENCE_0, df_fasta["seq"][0])
        self.assertEqual(TEST_FASTA_1_ID_1, df_fasta["id"][0])
        self.assertEqual(TEST_FASTA_1_TM_1, df_fasta["TM"][1])
        
    def test_ids(self):
        self.assertRaises(ValueError, read_fasta, TEST_FASTA_2)


class TestCreateEmbeding(unittest.TestCase):
    def test_embeding(self): 
        df_fasta = read_fasta(TEST_FASTA_3)
        df_fasta = creat_embedings(df_fasta)
        self.assertEqual(TEST_FASTA_3_ROWS, len(df_fasta["Embedding"]))
        self.assertEqual(TEST_FASTA_3_COLUMNS, len(df_fasta.keys()))
        self.assertEqual(TEST_FASTA_3_EMBEDDING_SIZE, len(df_fasta["Embedding"][0]))
        
    
