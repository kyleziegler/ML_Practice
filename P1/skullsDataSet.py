# Kyle Ziegler 
# ML Practice

from IPython.core.display import HTML
import numpy as np
import pandas 
from sklearn.neighbors import KNeighborsClassifier

HTML("""
    <style type="text/css">
    #ans:hover { 
        background-color: black; 
    }
    #ans {  
        padding: 6px; 
        background-color: white; 
        border: green 2px solid; 
        font-weight: bold; 
    }
    </style>
""")

skullsDataSet = pandas.read_csv("skulls.csv",delimiter = ",")
print(skullsDataSet)
print(type(skullsDataSet))

