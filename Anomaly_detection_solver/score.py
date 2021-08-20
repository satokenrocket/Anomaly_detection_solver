import urllib.request
import numpy as np
import keras
from tensorflow.keras.utils import plot_model

from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA

from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM



def get_score_LOF(model, x_train, x_test,label_x_test, n_neighbors):
    model_s = Model(inputs=model.input,outputs=model.layers[-2].output)
    train = model_s.predict(x_train, batch_size=1)
    test = model_s.predict(x_test, batch_size=1)

    ms = MinMaxScaler()
    train = ms.fit_transform(train)
    test = ms.transform(test)

    lof = LocalOutlierFactor(n_neighbors=5, novelty=True, contamination=0.1)
    lof.fit(train)
    Z = -lof._decision_function(test)

    return Z







