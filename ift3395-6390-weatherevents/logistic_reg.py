import numpy as np
import pandas as pd
from scipy import sparse

def sigmoid(x) :
    return 1/(1+np.exp(-x))

