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

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE





# # t-SNE_before\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def before(df_test, df_x, num_true_anoms, anomaly_sequences_start, anomaly_sequences_end,path):
    d = TSNE(n_components = 2).fit_transform(df_test) 
    fig = plt.figure(figsize=(13,7))

    for i in range(num_true_anoms):

        d1 = d[anomaly_sequences_start[i]:anomaly_sequences_end[i]]
        plt.scatter(d1[:,0], d1[:,1],c="red")

    for i in range(num_true_anoms):
        d = np.delete(d, np.s_[anomaly_sequences_start[i]:anomaly_sequences_end[i]],0)
    d2 =d
    print(d2.shape)


    
    plt.scatter(d2[:,0], d2[:,1], c="blue")
    plt.title('test_data_before.png', fontsize=22)
    fig.savefig(path+"/test_data_before.png")
# ////////////////////////////////////////////////////////////////////////
# # # # t-SNE 3D Test\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # test =  pd.DataFrame(df_test)
    # d = TSNE(n_components = 3,perplexity=35.0).fit_transform(test) 
    # fig = plt.figure(figsize=(10, 10)).gca(projection='3d')


    # for i in range(num_true_anoms):
    #     print(anomaly_sequences_start[i])
    #     d1 = d[anomaly_sequences_start[i]:anomaly_sequences_end[i]]
    
    #     fig.scatter(d1[:,0], d1[:,1], d1[:,2],c="red")

    # for i in range(num_true_anoms):
    #     d = np.delete(d, np.s_[anomaly_sequences_start[i]:anomaly_sequences_end[i]],0)
    # d2 =d
    # fig.scatter(d2[:,0], d2[:,1],d2[:,2], c="blue")
    # # buf = BytesIO()
    # # fig.figure.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    
    # # images = [Image.open(angle) for angle in range(360)]
    # # images[0].save('output.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

    # # fig.savefig("test_data_after.png")
    # plt.show()
    # fig.savefig(path+"/test_data_before.png")
 # # # t-SNE 3D Test\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
















# t-SNE_after////////////////////////////////////////////////////////////////////
    
def after(test,train,num_true_anoms,anomaly_sequences_start,anomaly_sequences_end,path):
    
    test =  pd.DataFrame(test)
    d = TSNE(n_components = 2).fit_transform(test) 
    fig = plt.figure(figsize=(13,7))


    for i in range(num_true_anoms):
        print(anomaly_sequences_start[i])
        d1 = d[anomaly_sequences_start[i]:anomaly_sequences_end[i]]
        plt.scatter(d1[:,0], d1[:,1],c="red")


    for i in range(num_true_anoms):
        d = np.delete(d, np.s_[anomaly_sequences_start[i]:anomaly_sequences_end[i]],0)
    d2 =d
    print(d2.shape)
    

    plt.scatter(d2[:,0], d2[:,1], c="blue")
    plt.title('test_data_after.png', fontsize=22)
    fig.savefig(path+"/test_data_after.png")
    # plt.show()
# ////////////////////////////////////////////////////////////////////////

# # # # t-SNE 3D Test\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # test =  pd.DataFrame(test)
    # d = TSNE(n_components = 3,perplexity=35.0).fit_transform(test) 
    # fig = plt.figure(figsize=(10, 10)).gca(projection='3d')

    # for i in range(num_true_anoms):
    #     print(anomaly_sequences_start[i])
    #     d1 = d[anomaly_sequences_start[i]:anomaly_sequences_end[i]]
    #     plt.scatter(d1[:,0], d1[:,1],c="red")


    # for i in range(num_true_anoms):
    #     d = np.delete(d, np.s_[anomaly_sequences_start[i]:anomaly_sequences_end[i]],0)
    # d2 =d
    # fig.scatter(d2[:,0], d2[:,1],d2[:,2], c="blue")
    # fig.savefig("test_data_after.png")
    # # fig.savefig("test_data_after.png")
    # plt.show()
    # fig.savefig(path+"/test_data_after.png")
# ////////////////////////////////////////////////////////////////////////



