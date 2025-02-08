import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']