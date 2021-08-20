import urllib.request
import numpy as np
import os
import keras
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
from tensorflow.keras.utils import plot_model



def picture(input_shikiichi,ijo,seijo,f,goto,path):

#recall-----------------
    plt.figure(figsize=(13,13))
    plt.title('Recall.png', fontsize=24)
    plt.scatter(input_shikiichi[:], ijo[:], c="red")
    plt.plot(input_shikiichi[:], ijo[:], c="red")
    plt.grid(True) 
    plt.xlabel("threshold", fontsize=16,fontname="MS Gothic")
    plt.ylabel("recall", fontsize=16,fontname="MS Gothic")
    plt.savefig(path+"/recall.png")
#------------------------------


#detection_ratio-----------------
    plt.figure(figsize=(13,13))
    plt.title('Detection_ratio.png', fontsize=24)
    plt.scatter(input_shikiichi[:], seijo[:], c="blue")
    plt.plot(input_shikiichi[:], seijo[:], c="blue")
    plt.grid(True) 
    plt.xlabel("threshold", fontsize=16,fontname="MS Gothic")
    plt.ylabel("detection_ratio",fontsize=16, fontname="MS Gothic")
    # plt.savefig(os.path.join(picture_path,"detection_rate.png"))
    plt.savefig(path+"/detection_ratio.png")
#------------------------------   



#break_even_point-----------------------
    plt.figure(figsize=(13,13))
   
    plt.xlim(1, 13)
    plt.ylim(0.0, 1.0)
    plt.title('Break_even_point.png', fontsize=24)
    fig, ax1 = plt.subplots( )
    plt.grid(True) 
    ax1.set_xlabel('threshold', fontsize=14,fontname="MS Gothic")
    ax1.set_ylabel('recall', color='red',fontsize=14, fontname="MS Gothic")
    ax1.scatter(input_shikiichi[:], ijo[:], c="red")
    ax1.plot(input_shikiichi[:], ijo[:], c="red")
    ax2 = ax1.twinx()  
    ax2.set_ylabel('detection_ratio', color='blue', fontname="MS Gothic")  
    ax2.scatter(input_shikiichi[:], seijo[:], c="blue")
    ax2.plot(input_shikiichi[:], seijo[:], c="blue")

    # plt.savefig(os.path.join(picture_path,"break_even_point.png"))
    plt.savefig(path+"/break_even_point.png")
#----------------------------------

#F-score-----------------
    plt.figure(figsize=(13,13))
    plt.title('F-score.png', fontsize=24)
    plt.scatter(input_shikiichi[:], f[:], c="green")
    plt.plot(input_shikiichi[:], f[:], c="green")
    plt.grid(True) 
    plt.xlabel("threshold", fontsize=20,fontname="MS Gothic")
    plt.ylabel("F-score",fontsize=20)
    # plt.savefig(os.path.join(picture_path,"Fscore.png"))
    plt.savefig(path+"/Fscore.png")
#------------------------------   


# #ROC曲線-----------------
    plt.figure(figsize=(13,13))
    plt.title('ROC.png', fontsize=24)
    plt.grid(True) 
    plt.scatter(goto[:], ijo[:], c="blue")
    plt.plot(goto[:], ijo[:], c="blue")
    plt.xlabel("False positive rate", fontsize=16,fontname="MS Gothic")
    plt.ylabel("True positive rate", fontsize=16,fontname="MS Gothic")
    # plt.savefig(os.path.join(picture_path,"ROC.png"))
    plt.savefig(path+"/ROC.png")
# #------------------------------   