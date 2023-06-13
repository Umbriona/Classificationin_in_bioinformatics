import os, sys
path_utils = os.path.dirname(os.getcwd())
sys.path.append(path_utils)
import unittest
import pandas as pd
from src.helper_functions import bins

ID_40     = "0 > x<= 40"
ID_40_60  = "40 > X >= 60"
ID_60_80  = "60 > X >= 80"
ID_80_100 = "80 > X >100"
ID_100    = "100 = X"

class TestBins(unittest.TestCase):
    def test_shape(self): 
        self.assertEqual(ID_40, bins(40))
        self.assertEqual(ID_40, bins(0))
        
        self.assertEqual(ID_40_60, bins(41))
        self.assertEqual(ID_40_60, bins(60))
        
        self.assertEqual(ID_60_80, bins(61))
        self.assertEqual(ID_60_80, bins(80))
        
        self.assertEqual(ID_80_100, bins(81))
        self.assertEqual(ID_80_100, bins(99))
        
        self.assertEqual(ID_100, bins(100))




