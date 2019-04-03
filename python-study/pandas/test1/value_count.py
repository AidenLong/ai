# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

ser1 = pd.Series(['a','b','c','a','d','c','b','c'])
print(ser1)
print(ser1.value_counts())
