import urllib.request
import numpy as np
import os
import keras
import SNE
import sco_picture
import csv
import score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pandas import DataFrame
import glob
from tensorflow.keras.utils import plot_model
from keras.utils import to_categorical
from keras.layers import Conv1D, Input, GlobalAveragePooling2D, GlobalAveragePooling1D, Dense, Dropout,LSTM, concatenate,  GRU, SimpleRNN,Permute
from keras.layers import Activation, BatchNormalization, Conv2D
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder




from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from keras.callbacks import EarlyStopping 


def main():
    # -----------------------------------------------------------------------------

    labeled_anomalies = "anomaly_sequences.csv"
    
    df = pd.read_csv(labeled_anomalies)
    data = pd.read_csv(labeled_anomalies).values.tolist()

    for num in range(len(data)):
        chan_id = data[num][0]
        num_values = data[num][1]
        num_true_anoms = data[num][2]
        
        
        path = '../output/'+chan_id
        if not os.path.exists(path):
            os.mkdir(path)
        
        

        # f_TEST_name = "../data/TEST/"+chan_id+".csv"
        # f_TRAIN_name = "../data/TRAIN/"+chan_id+".csv"

        f_TEST_name = "../data/TEST/"+chan_id+".csv"
        f_TRAIN_name = "../data/TRAIN/"+chan_id+".csv"
        input_shikiichi = "input_shikiichi.txt"

        

        d = TSNE()  

        # データ抽出
        data_TEST =  np.loadtxt(f_TEST_name)
        data_TRAIN =  np.loadtxt(f_TRAIN_name)

        
        T1 = len(data_TEST)
        T2 = len(data_TRAIN)

        # parameter 
        classes = 2 #k-meansのクラス数
        n_neighbors = 5 #LOFの近傍点の数
        w = 100 # 抽出窓の幅
        batch_size = 32
        epochs = 100
        NUM_CELLS = 8
        num_values = num_values-w


        x, x_test = [], []

        for i in range(w, T2):
            x.append(data_TRAIN[i-w: i])

        for i in range(w, T1):
            x_test.append(data_TEST[i-w: i])

        x = np.array(x)
        x_test = np.array(x_test)
        label_x_test = np.zeros(len(x_test))
        label_x_test[0:len(x_test)] = 1

        anomaly_sequences_start = []
        anomaly_sequences_end = []
        for i in range(num_true_anoms):
            anomaly_sequences_start.append(int(data[num][i*2+3]))
            anomaly_sequences_end.append(int(data[num][i*2+4]))
            label_x_test[int(data[num][i*2+3]):int(data[num][i*2+4])] = -1


        # t-SNEによる可視化----------------------
        df_test =  pd.DataFrame(x_test)
        df_x = pd.DataFrame(x)

        SNE_before = SNE.before(df_test,df_x,num_true_anoms,anomaly_sequences_start,anomaly_sequences_end,path)
        # ----------------------------------------


        # k-meansによる正常データのラベリング-----------------------------
        
        y_train = KMeans(n_clusters=classes).fit_predict(x)
        Y_train = to_categorical(y_train)

        #---------------------------------------------------
        

        # trainで定義したモデルによる学習-----------------------------
        X_train = np.expand_dims(x, axis=1)
        X_test = np.expand_dims(x_test, axis=1)
        model = train(X_train, Y_train, w, batch_size, epochs, NUM_CELLS )



        model_s = Model(inputs=model.input,outputs=model.layers[-2].output)
        x_train = model_s.predict(X_train, batch_size=1)
        x_test = model_s.predict(X_test, batch_size=1)

        x_train = x_train.reshape((len(X_train),-1))
        x_test = x_test.reshape((len(X_test),-1))
        #---------------------------------------------------


        # t-SNEによる可視化---------------------
        SNE_after = SNE.after(x_test,x_train,num_true_anoms,anomaly_sequences_start,anomaly_sequences_end,path)
        # ----------------------------------------

       
        text_file = open(path+"/input2.txt", "wt")
        get_score = score.get_score_LOF(model, X_train, X_test,label_x_test, n_neighbors)
       

    #閾値の算出---------------------------------------------------------------
        np.savetxt(path+'/get_score',get_score)
        h = (max(get_score)-min(get_score))/10
        input_shikiichi=[]

        input_shikiichi.append(min(get_score))
        text_file.write(str(x))

        x=0
        for i in range(10):
            x = min(get_score)+(h*i)
            input_shikiichi.append(x)
            text_file.write(str(x))
        
        input_shikiichi.append(max(get_score)+1)
        text_file.write(str(x))
    #-----------------------------------------------------------------
        
        for  i in range(num_true_anoms):
            anomaly_length = anomaly_sequences_end[i]-anomaly_sequences_start[i]


   # 性能評価-----------------------------------------------------------------
        #recall
        ijo = []
        ijo_plot = []
        for i in input_shikiichi:
            kt=0
            k=0
            for j in range(w, T1):
                if  get_score[j-w] <= 0:
                    kt=kt+1
                if get_score[j-w] >=  i and label_x_test[j-w] == -1:
                    k=k+1
            d=k/(anomaly_length)
            ijo.append(d)
            # ijo_plot.append(k)
        np.savetxt(path+'/ijo.txt', ijo)

    
        # detection_ratio
        seijo = []
        goto = []
        for i in input_shikiichi:
            kt=0
            k=0
            for j in range(w, T1):
                if get_score[j-w] <  i and label_x_test[j-w] == 1:
                    k=k+1
            b=k/(len(x_test)-(anomaly_length)) 
            c=1.0-b
            seijo.append(b)
            goto.append(c)
        np.savetxt(path+'/seijo.txt', seijo) 
        np.savetxt(path+'/goto.txt', goto)   


        # #F-score
        f = []
        for i in range(len(seijo)):
            k = (1.25*seijo[i]*ijo[i])/(0.25*seijo[i]+ijo[i]) #F0.5 score
            # k = (2*seijo[i]*ijo[i])/(seijo[i]+ijo[i])  #F1 score
            f.append(k)
        np.savetxt(path+'/F_score.txt', f)  
    #-----------------------------------------------------------------


        #各種精度計算のグラフ-----------------------------------------
        sco = sco_picture.picture(input_shikiichi,ijo,seijo,f,goto,path)
        #-----------------------------------------------------------

        

        # 異常スコアの可視化--------------------------------------------------------

        plt.figure(figsize=(12,12))
        plt.subplot(2,1,1)
        plt.plot(data_TEST)
        for  i in range(num_true_anoms):
            x2 = np.arange(anomaly_sequences_start[i],anomaly_sequences_end[i])
            y1=min(data_TEST)
            y2=max(data_TEST)
            plt.fill_between(x2, y1, y2, facecolor='r', alpha=.3)
        plt.legend()
        plt.grid(True) 

        plt.subplot(2,1,2)
        plt.plot(np.arange(num_values),get_score, label="score")
        for  i in range(num_true_anoms):
            x2 = np.arange(anomaly_sequences_start[i],anomaly_sequences_end[i])
            y1=min(get_score)
            y2=max(get_score)
            plt.fill_between(x2, y1, y2, facecolor='r', alpha=.3)
        plt.grid(True) 
        plt.savefig(path+"/score.png")

        # ----------------------------------------------------





def train(X_tra, Y_tra, w, batch_size, epochs, NUM_CELLS):
# ここに試したい深層学習のモデルを構築--------------------------------
# # alpha = 5
    ip = Input(shape=X_tra.shape[1:])
    x = LSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    # Y_tra.shape[1]=5
    out = Dense(Y_tra.shape[1], activation='softmax')(x)
  
    model = Model(ip, out)
    model.summary()


    model.compile(loss='categorical_crossentropy',
                  optimizer = Adam(lr=0.0001, amsgrad=True),
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=0, verbose=1) 
    # history = model.fit(X_tra, Y_tra, batch_size=32, epochs=50, validation_data=(X_tra, Y_tra), verbose=True)
    # history = model.fit(X_tra, Y_tra, batch_size = batch_size, epochs = epochs, verbose=True)
    history = model.fit(X_tra, Y_tra, batch_size = batch_size, epochs = epochs, verbose=True)
# 
#-------------------------------------------------------------------  

    return model






if __name__ =='__main__':
    main()


