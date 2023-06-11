import os, sys
path_utils = os.path.dirname(os.getcwd())
sys.path.append(path_utils)
import unittest
import pandas as pd
from src.make_embedings import read_fasta


# TEST of test1.fasta
TEST_FASTA_1 = "test1.fasta"
TEST_FASTA_2 = "test1.fasta"
TEST_FASTA_ROWS = 2
TEST_FASTA_COLUMNS = 3
TEST_FASTA_SEQUENCE_0 = "SANDRA"
TEST_FASTA_ID_1 = "sequence1"
TEST_FASTA_TM_1 = 90.0






class TestReadFasta(unittest.TestCase):
    def test_shape(self): 
        df_fasta = read_fasta(TEST_FASTA_1)
        self.assertEqual(TEST_FASTA_ROWS, df_fasta.shape[0])
        self.assertEqual(TEST_FASTA_COLUMNS, df_fasta.shape[1])
        
    def test_read(self):
        df_fasta = read_fasta(TEST_FASTA_1)
        self.assertEqual(TEST_FASTA_SEQUENCE_0, df_fasta["sequence"][0].values)
        self.assertEqual(TEST_FASTA_ID_1, df_fasta["id"][1].values)
        self.assertEqual(TEST_FASTA_TM_1, df_fasta["ogt"][1].values)
        
    def test_ids(self):
        self.assertRaises(ValueError, read_fasta, TEST_FASTA_2)

        
