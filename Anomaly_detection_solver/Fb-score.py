import urllib.request
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tensorflow.keras.utils import plot_model

from keras.utils import to_categorical
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from keras.layers import Activation, BatchNormalization, Conv2D
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE


f_TEST_name = "../data/E-8_TEST.txt"
f_TRAIN_name = "../data/E-8_TRAIN.txt"

d = TSNE()
# データ抽出
data_TEST =  np.loadtxt(f_TEST_name)
data_TRAIN =  np.loadtxt(f_TRAIN_name)

m_dist_th = np.zeros()