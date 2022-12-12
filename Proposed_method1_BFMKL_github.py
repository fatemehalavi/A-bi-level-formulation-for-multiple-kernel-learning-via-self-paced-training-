# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:56:13 2021

@author: Asus
"""

"""
In The Name of Allah

My thesis code

@author: Alavi
"""
import sklearn.datasets
import numpy as np
from numpy import array
import math
from sklearn import preprocessing
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
import cvxopt
from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, cosine_similarity, euclidean_distances, chi2_kernel
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp
from sklearn.cluster import KMeans
#from MKLpy.preprocessing import kernel_normalization, tracenorm, kernel_centering
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from scipy.optimize import minimize
import random
from scipy.io import loadmat
from scipy.io import savemat
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale
from qpsolvers import solve_qp
import math

#from sklearn.metrics.pairwise import euclidean_distances
solvers.options['show_progress'] = False
#from qpsolvers import solve_qp
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Read all text of file
def Create_MK_tr(data):
    # Gaussian Kernel
    # min_max_scaler = preprocessing.MinMaxScaler()
    # normalized = min_max_scaler.fit(data)
    # data = normalized.transform(data)
    
    p_list = np.arange(-15,16)
    # d_list = np.arange(1,20)
    t_list = list(np.power(2.0, p_list))
    n_smpl = data.shape[0]
    H = np.identity(n_smpl) - (1.0/n_smpl) * np.ones((n_smpl,n_smpl))
    
    # num_kernels = p_list.shape[0] + d_list.shape[0]
    num_kernels = p_list.shape[0]
    MK_total = np.zeros((n_smpl,n_smpl, num_kernels))
    idx = 0
    
    for t in t_list:
        kernel = rbf_kernel( data, data, t)
        kernel = np.matmul(np.matmul(H, kernel), H)
        # trn = np.trace(kernel) / n_smpl
        # kernel /= trn
        # kernel_dg = np.diag(kernel)
        # kernel = kernel/np.sqrt( np.matmul( np.array( kernel_dg, ndmin=2).T, np.array( kernel_dg, ndmin=2)))
        
        MK_total[:,:,idx] = kernel
        idx += 1    
    
    # for d in d_list:
    #     kernel = polynomial_kernel( data, data, d, 1)
    #     kernel = np.matmul(np.matmul(H, kernel), H)
    #     kernel_dg = np.diag(kernel)
    #     kernel = kernel/np.sqrt( np.matmul( np.array( kernel_dg, ndmin=2).T, np.array( kernel_dg, ndmin=2)))
        
    #     MK_total[:,:,idx] = kernel
    #     idx += 1
        
    return MK_total


def Create_MK_te_normal3(data, idx1, idx2):
    # Gaussian Kernel
    # min_max_scaler = preprocessing.MinMaxScaler()
    # normalized = min_max_scaler.fit(data)
    # data = normalized.transform(data)
    
    p_list = np.arange(-15,16)
    # d_list = np.arange(1,20)
    t_list = list(np.power(2.0, p_list))
    n_smpl = data.shape[0]
    n1 = idx1.shape[0]
    n2 = idx2.shape[0]
    H1 = np.identity(n1) - (1.0/n1) * np.ones((n1,n1))
    H2 = np.identity(n2) - (1.0/n2) * np.ones((n2,n2))
    # H = np.identity(n_smpl) - (1.0/n_smpl) * np.ones((n_smpl,n_smpl))
    
    # num_kernels = p_list.shape[0] + d_list.shape[0]
    num_kernels = p_list.shape[0]
    MK_total = np.zeros((n1, n2, num_kernels))
    idx = 0
    
    for t in t_list:
        kernel = rbf_kernel( data[idx1,], data[idx2,], t)
        kernel = np.matmul(np.matmul(H1, kernel), H2)
        # trn = np.trace(rbf_kernel( data[idx1,], data[idx1,], t)) / n1
        # kernel /= trn
        # kernel_dg1 = np.diag(rbf_kernel( data[idx1,], data[idx1,], t))
        # kernel_dg2 = np.diag(rbf_kernel( data[idx2,], data[idx2,], t))
        
        # kernel_dg = np.diag(kernel)
        # kernel = kernel/np.sqrt( np.matmul( np.array( kernel_dg1, ndmin=2).T, np.array( kernel_dg2, ndmin=2)))
        
        MK_total[:,:,idx] = kernel
        idx += 1    
    
    # for d in d_list:
    #     kernel = polynomial_kernel( data, data, d, 1)
    #     kernel = np.matmul(np.matmul(H, kernel), H)
    #     kernel_dg = np.diag(kernel)
    #     kernel = kernel/np.sqrt( np.matmul( np.array( kernel_dg, ndmin=2).T, np.array( kernel_dg, ndmin=2)))
        
    #     MK_total[:,:,idx] = kernel
        # idx += 1
    return MK_total
       
def Read_DataSet(dataset_type, dataset_name):
    if dataset_type =='clf_noise':
        dataset_type = 'clf'
    
    if dataset_type == 'ovr':
                
        if dataset_name == 'glass':
            data, label = load_svmlight_file('dataset/multi-class/glass/glass.scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/glass.mat", mdic)
            return data, label
        
        if dataset_name == 'iris':
            data, label = load_svmlight_file('dataset/multi-class/iris/iris.scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/iris.mat", mdic)
            return data, label
        
        if dataset_name == 'satimage':
            data, label = load_svmlight_file('dataset/multi-class/satimage/satimage.scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/satimage.mat", mdic)
            return data, label
                    
        if dataset_name == 'segment':
            data, label = load_svmlight_file('dataset/multi-class/segment/segment.scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/segment.mat", mdic)
            return data, label
                                
        if dataset_name == 'sensorless':
            data, label = load_svmlight_file('dataset/multi-class/sensorless/sensorless')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/sensorless.mat", mdic)
            return data, label
                                                        
        if dataset_name == 'dna':
            data, label = load_svmlight_file('dataset/multi-class/dna/dna.scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/dna.mat", mdic)
            return data, label
                                                                    
        if dataset_name == 'letter':
            data, label = load_svmlight_file('dataset/multi-class/letter/letter.scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/letter.mat", mdic)
            return data, label
                                                                                
        if dataset_name == 'svmguide2':
            data, label = load_svmlight_file('dataset/multi-class/svmguide2/svmguide2')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/svmguide2.mat", mdic)
            return data, label
                                                                                
        if dataset_name == 'vehicle':
            data, label = load_svmlight_file('dataset/multi-class/vehicle/vehicle.scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/vehicle.mat", mdic)
            return data, label
                                                                                            
        if dataset_name == 'wine':
            data, label = load_svmlight_file('dataset/multi-class/wine/wine.scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/wine.mat", mdic)
            return data, label
                
        if dataset_name == 'pendigits':
            X_train, y_train = load_svmlight_file('dataset/multi-class/pendigits/pendigits')
            X_train = X_train.toarray()
            X_test, y_test = load_svmlight_file('dataset/multi-class/pendigits/pendigits')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/pendigits.mat", mdic)
            return data, label
                                                                                                        
        if dataset_name == 'svmguide4':
            X_train, y_train = load_svmlight_file('dataset/multi-class/svmguide4/svmguide4')
            X_train = X_train.toarray()
            X_test, y_test = load_svmlight_file('dataset/multi-class/svmguide4/svmguide4.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/svmguide4.mat", mdic)
            return data, label 
                                                                                                                    
        if dataset_name == 'vowel':
            X_train, y_train = load_svmlight_file('dataset/multi-class/vowel/vowel')
            X_train = X_train.toarray()
            X_test, y_test = load_svmlight_file('dataset/multi-class/vowel/vowel.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/vowel.mat", mdic)
            return data, label

        
    if dataset_type == 'clf':

        if dataset_name == 'breast-cancer':
            myfile = open('dataset/clf/breast-cancer/breast-cancer-wisconsin.data','r')
            data = []
            label = []
            for line in myfile:
                sample = []
                features = line.split(',')
                for fea in features[1:-1]:
                    # missing value
                    if fea == '?':
                        sample.append(-1000)
                    else:
                        sample.append(float(fea))
                data.append(sample)
                # label 2 , 4
                if int(features[-1]) == 2:
                    label.append(-1)
                else:
                    label.append(1)
            myfile.close()
            data = np.array(data)
            for i in range(data.shape[0]):
               data[i, np.where( data[i,:] == -1000)] = np.mean(data[i,np.where( data[i,:] != -1000 )])
            data = np.array(data)
            label = np.array(label)
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/breast-cancer.mat", mdic)
            # MK_total = Create_MK(data)
            # Create_MK_Save(data, label, 'dataset/clf/breast-cancer/')
            # return MK_total, label
            return data, label
        
        if dataset_name == 'climate':
            myfile = open('dataset/clf/climate/pop_failures.dat','r')
            lines = myfile.readlines()
            myfile.close()

            data = []
            label = []
            for line in lines:
                sample = []
                features = line.split()
                for fea in features[2:-1]:
                    sample.append(float(fea))
                data.append(sample)
                # label 0,1
                if int(features[-1]) == 0:
                    label.append(-1)
                else:
                    label.append(1)
            data = np.array(data)
            label = np.array(label)
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/climate.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/climate/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'fertility':
            myfile = open('dataset/clf/fertility/fertility_Diagnosis.txt','r')
            lines = myfile.readlines()
            myfile.close()

            data = []
            label = []
            for line in lines:
                sample = []
                features = line.split(',')
                for fea in features[:-1]:
                    sample.append(float(fea))
                data.append(sample)
                # label 0,1
                if 'N' in features[-1]:
                    label.append(-1)
                else:
                    label.append(1)
            data = np.array(data)
            label = np.array(label)
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/fertility.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/fertility/')
            return data, label
            
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'diabetic':
            myfile = open('dataset/clf/diabetic/messidor_features.arff.txt','r')
            lines = myfile.readlines()
            myfile.close()

            data = []
            label = []
            for line in lines:
                sample = []
                features = line.split(',')
                for fea in features[:-1]:
                    sample.append(float(fea))
                data.append(sample)
                # label 0,1
                if int(features[-1]) == 0:
                    label.append(-1)
                else:
                    label.append(1)
            data = np.array(data)
            label = np.array(label)
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/diabetic.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/diabetic/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'EEGEyeState':
            myfile = open('dataset/clf/EEG Eye State/EEGEyeState.txt','r')
            lines = myfile.readlines()
            myfile.close()

            data = []
            label = []
            for line in lines:
                sample = []
                features = line.split(',')
                for fea in features[:-1]:
                    sample.append(float(fea))
                data.append(sample)
                # label 0,1
                if int(features[-1]) == 0:
                    label.append(-1)
                else:
                    label.append(1)
            data = np.array(data)
            label = np.array(label)
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/EEGEyeState.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/EEG Eye State/')
            return data, label
            
            # MK_total = Create_MK(data)
            # return MK_total, label
        
        if dataset_name == 'haberman':
            myfile = open('dataset/clf/haberman/haberman.data','r')
            lines = myfile.readlines()
            myfile.close()

            data = []
            label = []
            for line in lines:
                sample = []
                features = line.split(',')
                for fea in features[:-1]:
                    sample.append(float(fea))
                data.append(sample)
                # label 2,1
                if int(features[-1]) == 2:
                    label.append(-1)
                else:
                    label.append(1)
            data = np.array(data)
            label = np.array(label)
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/haberman.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/haberman/')
            return data, label
            
            # MK_total = Create_MK(data)
            # return MK_total, label
        
        if dataset_name == 'heart':
            data, label = load_svmlight_file('dataset/clf/heart/heart')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/heart.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/heart/')
            return data, label
            
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'ionosphere':
            data, label = load_svmlight_file('dataset/clf/Ionosphere/ionosphere_scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/ionosphere.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/Ionosphere/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label
        
        if dataset_name == 'sonar':
            data, label = load_svmlight_file('dataset/clf/sonar/sonar_scale')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/sonar.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/sonar/')
            return data, label
            
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'spambase':
            myfile = open('dataset/clf/spambase/spambase.data','r')
            lines = myfile.readlines()
            myfile.close()

            data = []
            label = []
            for line in lines:
                sample = []
                features = line.split(',')
                for fea in features[:-1]:
                    sample.append(float(fea))
                data.append(sample)
                # label 0,1
                if int( features[-1]) == 0 :
                    label.append(-1)
                else:
                    label.append(1)
            data = np.array(data)
            label = np.array(label)
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/spambase.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/spambase/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label
            
        if dataset_name == 'a1a':
            X_train, y_train = load_svmlight_file('dataset/clf/a1a/a1a')
            X_train = X_train.toarray()
            
    
            X_test, y_test = load_svmlight_file('dataset/clf/a1a/a1a.t')
            X_test = X_test.toarray()
            
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))))
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a1a.mat", mdic)
            return data, label
            
            # Create_MK_Save(data, label, 'dataset/clf/a1a/')
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'a2a':
            X_train, y_train = load_svmlight_file('dataset/clf/a2a/a2a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/a2a/a2a.t')
            X_test = X_test.toarray()
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))))
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a2a.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/a2a/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'a3a':

            X_train, y_train = load_svmlight_file('dataset/clf/a3a/a3a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/a3a/a3a.t')
            X_test = X_test.toarray()
            
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))))
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a3a.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/a2a/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'a4a':

            X_train, y_train = load_svmlight_file('dataset/clf/a4a/a4a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/a4a/a4a.t')
            X_test = X_test.toarray()
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))))
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a4a.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/a2a/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'a5a':

            X_train, y_train = load_svmlight_file('dataset/clf/a5a/a5a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/a5a/a5a.t')
            X_test = X_test.toarray()
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))))
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a5a.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/a2a/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'a6a':

            X_train, y_train = load_svmlight_file('dataset/clf/a6a/a6a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/a6a/a6a.t')
            X_test = X_test.toarray()
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))))
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a6a.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/a2a/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'a7a':

            X_train, y_train = load_svmlight_file('dataset/clf/a7a/a7a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/a7a/a7a.t')
            X_test = X_test.toarray()
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))))
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a7a.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/a2a/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label


        if dataset_name == 'a8a':

            X_train, y_train = load_svmlight_file('dataset/clf/a8a/a8a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/a8a/a8a.t')
            X_test = X_test.toarray()
            X_test = np.hstack((X_test, np.zeros((X_test.shape[0], X_train.shape[1]-X_test.shape[1]))))
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a8a.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/a2a/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'a9a':

            X_train, y_train = load_svmlight_file('dataset/clf/a9a/a9a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/a9a/a9a.t')
            X_test = X_test.toarray()
            X_test = np.hstack((X_test, np.zeros((X_test.shape[0], X_train.shape[1]-X_test.shape[1]))))
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/a9a.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/a2a/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label

        if dataset_name == 'australian':
            data, label = load_svmlight_file('dataset/clf/australian/australian')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/australian.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/australian/')
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label


        if dataset_name == 'diabetes':
            data, label = load_svmlight_file('dataset/clf/diabetes/diabetes')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/diabetes.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/diabetes/')
            return data, label

        if dataset_name == 'fourclass':
            data, label = load_svmlight_file('dataset/clf/fourclass/fourclass')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/fourclass.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/fourclass/')
            # MK_total = Create_MK(data)
            # return MK_total, label
            return data, label

        if dataset_name == 'german':
            data, label = load_svmlight_file('dataset/clf/german/german.numer')
            data = data.toarray()
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/german.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/german/')
            # MK_total = Create_MK(data)
            # return MK_total, label
            return data, label
           
        if dataset_name == 'phishing':
            data, label = load_svmlight_file('dataset/clf/phishing/phishing')
            data = data.toarray()
            label[np.where( label == 0)] = -1
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/phishing.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/phishing/')
            # MK_total = Create_MK(data)
            # return MK_total, label
            return data, label
            
    
        if dataset_name == 'guide1-t':
            X_train, y_train = load_svmlight_file('dataset/clf/guide1-t/svmguide1')
            X_train = X_train.toarray()
            y_train[np.where( y_train == 0)] = -1
    
            X_test, y_test = load_svmlight_file('dataset/clf/guide1-t/svmguide1.t')
            X_test = X_test.toarray()
            y_test[np.where( y_test == 0)] = -1
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/guide1-t.mat", mdic)
            # Create_MK_Save(data, label, 'dataset/clf/guide1-t/')
            # MK_total = Create_MK(data)
            # return MK_total, label
            return data, label
    
    
        if dataset_name == 'monks1':
            myfile = open('dataset/clf/monks1/monks-1.train','r')
            lines = myfile.readlines()
            myfile.close()
    
            X_train = []
            y_train = []
            for line in lines:
                sample = []
                features = line.split()
                for fea in features[1:-1]:
                    sample.append(float(fea))
                X_train.append(sample)
                # label 0,1
                if int(features[0]) == 0 :
                    y_train.append(-1)
                else:
                    y_train.append(1)
    
            myfile = open('dataset/clf/monks1/monks-1.test','r')
            lines = myfile.readlines()
            myfile.close()
            X_test = []
            y_test = []
            for line in lines:
                sample = []
                features = line.split()
                for fea in features[1:-1]:
                    sample.append(float(fea))
                X_test.append(sample)
                # label 0,1
                if int(features[0]) == 0 :
                    y_test.append(-1)
                else:
                    y_test.append(1)
    
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/monks1.mat", mdic)
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/monks1/')
            return data, label
    
    
        if dataset_name == 'monks2':
            myfile = open('dataset/clf/monks2/monks-2.train','r')
            lines = myfile.readlines()
            myfile.close()
    
            X_train = []
            y_train = []
            for line in lines:
                sample = []
                features = line.split()
                for fea in features[1:-1]:
                    sample.append(float(fea))
                X_train.append(sample)
                # label 0,1
                if int(features[0]) == 0 :
                    y_train.append(-1)
                else:
                    y_train.append(1)
    
            myfile = open('dataset/clf/monks2/monks-2.test','r')
            lines = myfile.readlines()
            myfile.close()
            X_test = []
            y_test = []
            for line in lines:
                sample = []
                features = line.split()
                for fea in features[1:-1]:
                    sample.append(float(fea))
                X_test.append(sample)
                # label 0,1
                if int(features[0]) == 0 :
                    y_test.append(-1)
                else:
                    y_test.append(1)
    
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/monks2.mat", mdic)
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/monks2/')
            return data, label
    
    
        if dataset_name == 'monks3':
            myfile = open('dataset/clf/monks3/monks-3.train','r')
            lines = myfile.readlines()
            myfile.close()
    
            X_train = []
            y_train = []
            for line in lines:
                sample = []
                features = line.split()
                for fea in features[1:-1]:
                    sample.append(float(fea))
                X_train.append(sample)
                # label 0,1
                if int(features[0]) == 0 :
                    y_train.append(-1)
                else:
                    y_train.append(1)
    
            myfile = open('dataset/clf/monks3/monks-3.test','r')
            lines = myfile.readlines()
            myfile.close()
            X_test = []
            y_test = []
            for line in lines:
                sample = []
                features = line.split()
                for fea in features[1:-1]:
                    sample.append(float(fea))
                X_test.append(sample)
                # label 0,1
                if int(features[0]) == 0 :
                    y_test.append(-1)
                else:
                    y_test.append(1)
    
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/monks3.mat", mdic)
            
            # Create_MK_Save(data, label, 'dataset/clf/monks3/')
            
            # MK_total = Create_MK(data)
            # return MK_total, label 
            return data, label
    
        if dataset_name == 'spect':
            myfile = open('dataset/clf/spect/SPECT.train','r')
            lines = myfile.readlines()
            myfile.close()
    
            X_train = []
            y_train = []
            for line in lines:
                sample = []
                features = line.split(',')
                for fea in features[1:]:
                    sample.append(float(fea))
                X_train.append(sample)
                # label 0,1
                if int(features[0]) == 0 :
                    y_train.append(-1)
                else:
                    y_train.append(1)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
    
            myfile = open('dataset/clf/spect/SPECT.test','r')
            lines = myfile.readlines()
            myfile.close()
    
            X_test = []
            y_test = []
            for line in lines:
                sample = []
                features = line.split(',')
                for fea in features[1:]:
                    sample.append(float(fea))
                X_test.append(sample)
                # label 0,1
                if int(features[0]) == 0 :
                    y_test.append(-1)
                else:
                    y_test.append(1)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/spect.mat", mdic)
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/spect/')
            return data, label
    
    
        if dataset_name == 'splice':
            X_train, y_train = load_svmlight_file('dataset/clf/splice/splice')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/splice/splice.t')
            X_test = X_test.toarray()
            
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/splice.mat", mdic)
            return data, label
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/splice/')
    
        if dataset_name == 'gisette':
            X_train, y_train = load_svmlight_file('dataset/clf/gisette/gisette_scale')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/gisette/gisette_scale.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # Create_MK_Save(data, label, 'dataset/clf/gisette/')
            # MK_total = Create_MK(data)
            # return MK_total, label
            return data, label
    
    
        if dataset_name == 'leukemia':
            X_train, y_train = load_svmlight_file('dataset/clf/leukemia/leu')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/leukemia/leu.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label
            return data, label
    
    
        if dataset_name == 'liver-disorders':
            X_train, y_train = load_svmlight_file('dataset/clf/liver-disorders/liver-disorders')
            X_train = X_train.toarray()
            y_train[np.where( y_train == 0)] = -1
    
            X_test, y_test = load_svmlight_file('dataset/clf/liver-disorders/liver-disorders.t')
            X_test = X_test.toarray()
            y_test[np.where( y_test == 0)] = -1
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/liver-disorders.mat", mdic)
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/liver-disorders/')
            return data, label
    
    
        if dataset_name == 'madelon':
            X_train, y_train = load_svmlight_file('dataset/clf/madelon/madelon')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/madelon/madelon.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 

            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/madelon/')
            return data, label
    
    
        if dataset_name == 'svmguide3':
            X_train, y_train = load_svmlight_file('dataset/clf/svmguide3/svmguide3')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/svmguide3/svmguide3.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test))
            # mdic = {"data": data, "label": label}
            # savemat("dataset/matfiles/svmguide3.mat", mdic)
            # MK_total = Create_MK(data)
            # return MK_total, label 
            # Create_MK_Save(data, label, 'dataset/clf/svmguide3/')
            return data, label
    
    
        if dataset_name == 'w1a':
            X_train, y_train = load_svmlight_file('dataset/clf/w1a/w1a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/w1a/w1a.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/w1a/')
            return data, label
    
    
        if dataset_name == 'w2a':
            X_train, y_train = load_svmlight_file('dataset/clf/w2a/w2a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/w2a/w2a.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/w2a/')
            return data, label
    
    
        if dataset_name == 'w3a':
            X_train, y_train = load_svmlight_file('dataset/clf/w3a/w3a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/w3a/w3a.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/w3a/')
            return data, label
    
    
        if dataset_name == 'w4a':
            X_train, y_train = load_svmlight_file('dataset/clf/w4a/w4a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/w4a/w4a.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/w4a/')
            return data, label
    
    
        if dataset_name == 'w5a':
            X_train, y_train = load_svmlight_file('dataset/clf/w5a/w5a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/w5a/w5a.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label 
            # Create_MK_Save(data, label, 'dataset/clf/w5a/')
            return data, label
    
    
        if dataset_name == 'w6a':
            X_train, y_train = load_svmlight_file('dataset/clf/w6a/w6a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/w6a/w6a.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/w6a/')
            return data, label
    
    
        if dataset_name == 'w7a':
            X_train, y_train = load_svmlight_file('dataset/clf/w7a/w7a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/w7a/w7a.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label 
            # Create_MK_Save(data, label, 'dataset/clf/w7a/')
            return data, label
    
    
        if dataset_name == 'w8a':
            X_train, y_train = load_svmlight_file('dataset/clf/w8a/w8a')
            X_train = X_train.toarray()
    
            X_test, y_test = load_svmlight_file('dataset/clf/w8a/w8a.t')
            X_test = X_test.toarray()
            data, label = np.vstack(( X_train,X_test)), np.concatenate((y_train, y_test)) 
            # MK_total = Create_MK(data)
            # return MK_total, label
            # Create_MK_Save(data, label, 'dataset/clf/w8a/')
            return data, label


def Obj_f(f, *args):
    S = args[0]
    gamma1 = args[1]
    t5 = 1/2 * np.sum(np.multiply(S, np.matmul( np.array(f, ndmin =2).T, np.array(f, ndmin =2)) ) )
    t4 = -gamma1 * np.sum(np.abs( f))
    # return  first_ord*w + lambda2/2 *t5 + t6
    return  t5 + t4

def Grad_f(f, *args):
    S = args[0]
    gamma1 = args[1]
    return  np.matmul(S, f) - gamma1* np.ones(f.shape[0])

def Global_Local_Corr_Parametric_Kernel_AGD_Proximal_S_Bilevel( MK_train, K_y, y_train, M, rho1, rho2, lambda3, neig_percent, type_norm, type_task):
          
    n_train = K_y.shape[0]
    num_kernels = MK_train.shape[2]
    max_itr = 5
    S = np.ones((n_train, n_train))
    epsilon_0 = 10**-4
    
    # Global Correlation
    w = np.ones(num_kernels) * (1/num_kernels)
    K_w = np.zeros((n_train, n_train))
    for k in range( num_kernels ):
        K_w += ( w[k] * MK_train[:,:,k] )
    
    rho2 = rho2 * np.linalg.norm(K_w)
    lambda_ = 1/np.linalg.norm(K_w)*10**-1
    
    num_nbrs = math.ceil( neig_percent * n_train)
    if num_nbrs < 5:
        num_nbrs = 5
    K_w_y = np.multiply(K_w, K_y)
    indices = np.zeros((n_train, num_nbrs), dtype=np.int)
    for inst in range(n_train):
        ranked = np.argsort(K_w[inst,:])
        largest_indices = ranked[::-1][:num_nbrs+1]
        largest_indices = np.delete(largest_indices, np.argwhere(largest_indices==inst))
        indices[inst,:] = largest_indices[:num_nbrs]
    
    gamma1 = 1
    P1 = matrix( K_w_y )
    q1 = matrix( -gamma1 * np.ones(n_train) )
    G1 = matrix(  np.vstack((-1 * np.identity( n_train ),  np.identity( n_train ))) )
    h1 = matrix(  np.hstack( ( np.zeros(n_train),  np.ones(n_train) )) )  
    f = np.array( solvers.qp(P1, q1, G1, h1)['x']).reshape(-1)
    f[f<= 9*10**-4] = 0
    corr_idxs = np.where(f==0)
    
    n_local = np.array(corr_idxs).ravel().shape[0] + 1           
    t1 = np.sum( np.multiply( np.multiply(K_w, S), K_y) )
    t2 = 0
    t4 = np.sum(np.sum(np.abs(S)**2, axis=-1)**(1./2))
    for idx1 in range( np.array(corr_idxs).ravel().shape[0]):
        idx = np.array(corr_idxs).ravel()[idx1]
        nbrs_tr = indices[idx,:]
        idexs = np.ix_(nbrs_tr, nbrs_tr)
        t2 += np.sum( np.multiply( np.multiply( K_w[idexs], S[idexs]), K_y[idexs]) )
        
    t3 = np.linalg.norm( S - np.ones((n_train, n_train)) )**2
    w_tmp = np.matmul( np.array( w, ndmin=2).T, np.array( w, ndmin=2))
    t5 = np.sum( np.multiply( M, w_tmp))
    obj_t = -1*((lambda_)*t1+ 1/n_local*t2)  + rho1/2*t3 + rho2*t4 + lambda3/2*t5
    
    max_outer_itr = 5
    outer_itr = 0
    while outer_itr< max_outer_itr:
        # Upper-level optimization
        n_local = np.array(corr_idxs).ravel().shape[0] + 1
        itr = 0
        isCont = True
        while isCont:
            K_w_y = np.multiply(K_w, K_y)
            Corr = np.array ( K_w_y )
            local_Corr = np.zeros((n_train, n_train))
            # InvMat = np.identity(n_train)
            l2_col = np.sum(np.abs(S)**2, axis=-1)**(1./2)
            InvMat = (2*rho1* l2_col)/(2*rho1* l2_col + rho2)
            InvMat = np.diag(InvMat)
            for idx1 in range( np.array(corr_idxs).ravel().shape[0]):
                idx = np.array(corr_idxs).ravel()[idx1]
    
                Nei_matrix = np.zeros((n_train, n_train))
                nbrs_tr = indices[idx,:]
                
                idexs = np.ix_(nbrs_tr, nbrs_tr)
                
                Nei_matrix[np.ix_(nbrs_tr,nbrs_tr)] = np.array( np.multiply( K_w[idexs], K_y[idexs]))

                local_Corr += Nei_matrix
            Corr = 1*(lambda_ * Corr + 1/n_local*local_Corr)
            
            S = np.matmul( ( 1/rho1* Corr + np.ones( (n_train, n_train))), InvMat )
            S = (S + S.T)/2
            if np.max(S)==float(0):
                print("S is zero")
            obj_new = []               
            for k in range(num_kernels):
                obj_new.append(-1 * (lambda_) * np.sum( np.multiply( np.multiply( MK_train[:,:,k] , K_y) , S)) )
                for idx1 in range( np.array(corr_idxs).ravel().shape[0]):
                    idx = np.array(corr_idxs).ravel()[idx1]
                    nbrs_tr = indices[idx,:]
                    idexs = np.ix_(nbrs_tr, nbrs_tr)
                    
                    obj_new[k] += -1/(n_local) * np.sum( np.multiply( np.multiply( MK_train[:,:,k][idexs], \
                                                              K_y[idexs]) , S[idexs]))
            obj_new = np.array( obj_new )
            P1 = matrix( lambda3 * M )
            q1 = matrix( obj_new )
            G1 =  matrix( -1 * np.identity( len(w) ) )
            h1 = matrix( np.zeros(len(w)) )
            w = np.array( solvers.qp(P1, q1, G1, h1)['x']).reshape(-1)
            w[w<0] = 0
            w = w / np.linalg.norm( w )
            
            K_w = np.zeros((n_train, n_train))
            for k in range( num_kernels ):
                K_w += ( w[k] * MK_train[:,:,k] )           
            t1 = np.sum( np.multiply( np.multiply(K_w, S), K_y) )
            t2 = 0 
            t4 = np.sum(np.sum(np.abs(S)**2, axis=-1)**(1./2))
            for idx1 in range( np.array(corr_idxs).ravel().shape[0]):
                idx = np.array(corr_idxs).ravel()[idx1]
                nbrs_tr = indices[idx,:]
                idexs = np.ix_(nbrs_tr, nbrs_tr)
                # t2 += (1-f[idx]) * np.sum( np.multiply( np.multiply(K_w[idexs], S[idexs]), K_y[idexs]) )
                t2 += np.sum( np.multiply( np.multiply( K_w[idexs], S[idexs]), K_y[idexs]) )
            t3 = np.linalg.norm( S - np.ones((n_train, n_train)) )**2
            w_tmp = np.matmul( np.array( w, ndmin=2).T, np.array( w, ndmin=2))
            t5 = np.sum( np.multiply( M, w_tmp))
            obj_t_1 = -1*( lambda_* t1+ 1/n_local*t2)  + rho1/2*t3 + rho2*t4 + lambda3/2*t5 
    
            if ( ( ( abs(obj_t - obj_t_1)/abs(obj_t_1)) <epsilon_0) and itr>1) or itr>max_itr:
                isCont = False            
            obj_t = obj_t_1            
            itr += 1
                    
        gamma1 /= 1.1
                
        P1 = matrix( K_w_y )
        q1 = matrix( -gamma1 * np.ones(n_train) )
        G1 = matrix(  np.vstack((-1 * np.identity( n_train ),  np.identity( n_train ))) )
        h1 = matrix(  np.hstack( ( np.zeros(n_train),  np.ones(n_train) )) )  
        f = np.array( solvers.qp(P1, q1, G1, h1)['x']).reshape(-1)
        f[f<= 9*10**-4] = 0
        corr_idxs = np.where(f==0)
        
              
        outer_itr += 1

    K_w = np.zeros((n_train, n_train))
    for k in range(num_kernels):
        K_w += ( w[k] * MK_train[:,:,k])

    if type_task == 'clf':
        clf = svm.SVC( kernel='precomputed')
        clf.fit( K_w, y_train)
        y_pred = clf.predict(K_w)
        acc_train = accuracy_score(y_train, y_pred) * 100
    else:
        clf = svm.SVC( kernel='precomputed', probability=True)
        clf.fit( K_w, y_train)
        y_pred = clf.predict(K_w)
        acc_train = accuracy_score(y_train, y_pred) * 100
    
    return clf, w, acc_train


def Print_func(dataset_name):
    
    print(dataset_name, "Simul_Global_Local_Corr_test: ",  np.mean(np.array(acc_list_simu_corr_test)), np.std(np.array(acc_list_simu_corr_test)))
    print(dataset_name, "Simul_Global_Local_Corr_train: ",  np.mean(np.array(acc_list_simu_corr_train)), np.std(np.array(acc_list_simu_corr_train)))       
    # print(dataset_name, "Simul_Global_Local_Corr_kta: ",  np.mean(np.array(acc_list_simu_corr_kta)), np.std(np.array(acc_list_simu_corr_kta)))
    print()

    myfile.write( dataset_name + "\tSimul_Global_Local_Corr_test: " + "mean: " + str(  np.mean(np.array(acc_list_simu_corr_test)) ) + "\tstd: "+ str( np.std(np.array(acc_list_simu_corr_test))) +'\n')
    myfile.write( dataset_name + "\tSimul_Global_Local_Corr_train: " + "mean: " + str(  np.mean(np.array(acc_list_simu_corr_train)) ) + "\tstd: "+ str( np.std(np.array(acc_list_simu_corr_train))) +'\n')   
    # myfile.write( dataset_name + "\tSimul_Global_Local_Corr_kta: " + "mean: " + str(  np.mean(np.array(acc_list_simu_corr_kta)) ) + "\tstd: "+ str( np.std(np.array(acc_list_simu_corr_kta))) +'\n')
    myfile.write("\n")
    
    myfile.write( dataset_name + "\tSimul_Global_Local_Corr_test: " +  str(  np.array(acc_list_simu_corr_test) ) + '\n')
    myfile.write( dataset_name + "\tSimul_Global_Local_Corr_train: " +  str(  np.array(acc_list_simu_corr_train) ) + '\n')
    # myfile.write( dataset_name + "\tSimul_Global_Local_Corr_kta: " +  str(  np.array(acc_list_simu_corr_kta) ) + '\n')
    myfile.write("\n")
    
    myfile.write("*"*50)


def Read_Indexs(dataset_name, times, n_train, n_test):
    myfile = open('dataset/indexes/' + dataset_name + '.txt','r')
    tr_indices = np.zeros((times, n_train), dtype=np.int)
    te_indices = np.zeros((times, n_test), dtype=np.int)
    itr = 0
    while itr < (times-1):
        itr = int ( myfile.readline() )
        # print(itr)
        train_ = myfile.readline()
        train_part = train_.split()
        for i in range(1,len(train_part)):
            tr_indices[itr,i-1] = int(train_part[i])    

        test_ = myfile.readline()
        test_part = test_.split()
        for i in range(1,len(test_part)):
            te_indices[itr,i-1] = int(test_part[i])

    return tr_indices, te_indices

# mn = cv
print ("salam")
myfile = open('FinalResult.txt','w')
dataset_type = 'clf'
# dataset_type = 'mcls'
type_norm = 'l2'


if dataset_type == 'clf':
    dataset_names = [ 'fertility', 'sonar', 'spect', 'haberman','liver-disorders',
                      'ionosphere', 'climate', 'monks3', 'monks1', 'monks2', 'australian',
                      'breast-cancer', 'diabetes', 'fourclass', 'german', 'diabetic', 
                      'svmguide3', 'splice', 'spambase',
                      'guide1-t', 'phishing', 'EEGEyeState' ]
    
    times = 10
    for dataset_name in dataset_names[2:3]:
        
        acc_list_simu_corr_test = []
        acc_list_simu_corr_train = []
        acc_list_simu_corr_kta = []
    
        myfile.write('*' * 100 +'\n')
        myfile.write('dataset_name:\t'+ dataset_name+'\n' )
        
        print(dataset_name)
        data , y = Read_DataSet(dataset_type, dataset_name)
        n_total = y.shape[0]
        print(n_total)
        n_train, n_test = int(n_total/2), n_total - int(n_total/2)
        
        train_indices, test_indices = Read_Indexs(dataset_name, times, n_train, n_test)
        
        for idx_time in range(times):
            myfile.write("iteration: "+str(idx_time) +'\n')
            train_idx, test_idx = train_indices[idx_time,:], test_indices[idx_time,:]
            
            min_max_scaler = preprocessing.MinMaxScaler()
            normalized = min_max_scaler.fit(data[train_idx,])
            data[train_idx,] = normalized.transform(data[train_idx,])
            data[test_idx,] = normalized.transform(data[test_idx,])
            
            rho1, rho2, lambda3, neig_percent = 2**5, 0.1, 2**0, 0.01
            
            MK_train = Create_MK_tr(data[ train_idx])
            num_kernels = MK_train.shape[2]
            K_y = np.matmul( np.array( y[train_idx], ndmin=2).T, np.array( y[train_idx], ndmin=2))

            M = np.zeros((num_kernels, num_kernels))        
            idx = 0
            tr_indices = np.ix_(train_idx, train_idx)
            for p1 in range(num_kernels):
                idx += 1
                for q1 in range(p1, num_kernels):
                    M[p1,q1] = np.sum( np.multiply( MK_train[:,:,p1], MK_train[:,:,q1]) )/( np.linalg.norm(MK_train[:,:,p1]) * np.linalg.norm(MK_train[:,:,q1]))
                    # M[p1,q1] = np.sum( np.multiply( MK_train[:,:,p1], MK_train[:,:,q1]) )
                    M[q1, p1] = M[p1,q1]
            
            clf_simu, w_simu, acc_train = Global_Local_Corr_Parametric_Kernel_AGD_Proximal_S_Bilevel( MK_train, K_y, y[train_idx], M, rho1, rho2, lambda3, neig_percent, type_norm, 'clf' )            
            acc_list_simu_corr_train.append( acc_train)
            
            del MK_train, K_y

            MK_test = (Create_MK_te_normal3(data, train_idx, test_idx))[clf_simu.support_,]            
                        
            K_w_test = np.zeros(( np.sum( clf_simu.n_support_), n_test))
            for k in range(num_kernels):
                K_w_test += ( w_simu[k] * MK_test[:,:,k])
            
            y_pred = np.sign( np.matmul(clf_simu.dual_coef_, K_w_test) + clf_simu.intercept_ )
            acc_test = accuracy_score(y[test_idx].ravel(), y_pred.ravel())*100
            acc_list_simu_corr_test.append( acc_test)
            myfile.write('rho1: '+ str(rho1)+'\trho2: '+ str(rho2)+'\tlambda3: '+str( lambda3)+'\tneig_percent: '+str(neig_percent) +'\n')
            myfile.write('test: '+ str( acc_test ) +'\ttrain: '+ str( acc_train)+'\n' )
            print('test:', acc_test,'train:', acc_train )
            
            del MK_test
        myfile.write('\n' )    
        Print_func(dataset_name)

if dataset_type == 'ovr':

    dataset_names = [ 'iris', 'wine', 'glass', 'svmguide2', 'svmguide4','vehicle', 
                     'vowel', 'dna', 'segment', 'satimage', 'pendigits', 'letter' ]
    
    # for dataset_name in dataset_names[0:]:
    #     data , y = Read_DataSet(dataset_type, dataset_name)
    #     Create_Indexs(dataset_name,y)
    # mn = cv
    # , 'pendigits', 'letter','sensorless'
    times = 10
    for dataset_name in dataset_names[0:-2]:
        
        acc_list_simu_corr_test = []
        acc_list_simu_corr_train = []
        acc_list_simu_corr_kta = []
    
        myfile.write('*' * 100 +'\n')
        myfile.write('dataset_name:\t'+ dataset_name+'\n' )
        
        print(dataset_name)
        data , y = Read_DataSet(dataset_type, dataset_name)
        # Create_Indexs(dataset_name,y)
        # mn = cv
        # y = Read_Label(dataset_type, dataset_name)
        n_total = y.shape[0]
        print(n_total)
        n_train, n_test = int(n_total/2), n_total -int(n_total/2)
        
        train_indices, test_indices = Read_Indexs(dataset_name, times, n_train, n_test)
        
        for idx_time in range(times):
            myfile.write("iteration: "+str(idx_time) +'\n')
            train_idx, test_idx = train_indices[idx_time,:], test_indices[idx_time,:]
            
            min_max_scaler = preprocessing.MinMaxScaler()
            normalized = min_max_scaler.fit(data[train_idx,])
            data[train_idx,] = normalized.transform(data[train_idx,])
            data[test_idx,] = normalized.transform(data[test_idx,])
            
            # lambda1, lambda2, lambda3, neig_percent = Parameter_Tune( data[train_idx,], y[train_idx], type_norm)
            lambda1, lambda2, lambda3, neig_percent = 2**5, 0.1, 2**0, 0.01
            MK_train = Create_MK_tr(data[train_idx,])
            num_kernels = MK_train.shape[2]
            # PEP
            # beta_eig = np.zeros(num_kernels)
            M = np.zeros((num_kernels, num_kernels))
            num_eig = 4
            # H = np.identity(n_train) - (1.0/n_train) * np.ones((n_train))
            
            idx = 0
            tr_indices = np.ix_(train_idx, train_idx)
            beta_eig = np.zeros(num_kernels)
            for p1 in range(num_kernels):
                # eigvals = np.linalg.eigvalsh(MK_train[:,:,p1])
                # beta_eig[idx] = np.sum(eigvals[-num_eig:])/np.trace( MK_train[:,:,p1])
                idx += 1
                for q1 in range(p1, num_kernels):
                    # M[p1,q1] = np.sum( np.multiply( MK_train[:,:,p1], MK_train[:,:,q1]) )/( np.linalg.norm(MK_train[:,:,p1]) * np.linalg.norm(MK_train[:,:,q1]))
                    M[p1,q1] = np.sum( np.multiply( MK_train[:,:,p1], MK_train[:,:,q1]) )
                    M[q1, p1] = M[p1,q1]
                        
            # print('lambda1, lambda2, lambda3, neig_percent', lambda1, lambda2, lambda3, neig_percent)
            # lambda1, lambda2, lambda3, neig_percent = 2, 2**3, 2, 0.05
                
            # MK_train, K_y = Read_DataSet_MK(dataset_type, dataset_name, train_idx)            
            predict_pr_te = []
            predict_pr_tr = []
            clf_list = []
            w_list = []
            labels = np.unique(y[train_idx])
            num_class = labels.shape[0]
            
            for lbl in labels:
                y_train_new = np.zeros(n_train)
                y_test_new = np.zeros(n_test)
                idx_pos = np.where(y[train_idx]==lbl)
                idx_neg = np.where(y[train_idx]!=lbl)
                y_train_new[idx_pos] = 1
                y_train_new[idx_neg] = -1
                
                idx_pos_te = np.where(y[test_idx]==lbl)
                idx_neg_te = np.where(y[test_idx]!=lbl)
                y_test_new[idx_pos_te] = 1
                y_test_new[idx_neg_te] = -1
            
                # yyt = np.matmul( np.array( y_train_new, ndmin=2).T, np.array( y_train_new, ndmin=2))            
                K_y = np.matmul( np.array( y_train_new, ndmin=2).T, np.array( y_train_new, ndmin=2))
                # clf_simu, w_simu, acc_train = Global_Local_Corr_Parametric_Kernel_AGD_Proximal_S_Bilevel( MK_train, K_y, y[train_idx], M, lambda1, lambda2, lambda3, neig_percent, type_norm, 'clf' )            

                clf, w_simu, acc_train = Global_Local_Corr_Parametric_Kernel_AGD_Proximal_S_Bilevel( MK_train, K_y, y_train_new, M, lambda1, lambda2, lambda3, neig_percent, type_norm, 'ovr' )            
                clf_list.append(clf)
                w_list.append(w_simu)
                # predict_pr_tr.append( (clf.predict_proba(K_w))[:,clf.classes_==1])        
                # predict_pr_te.append( (clf.predict_proba(K_w_test.T))[:,clf.classes_==1])  
        
                # acc_list_simu_corr_train.append( acc_train)
        
            for idx_clf in range(labels.shape[0]):
                clf = clf_list[idx_clf]
                K_w = np.zeros((n_train, n_train))
                w_simu = w_list[idx_clf]
                for k in range(num_kernels):
                    K_w += ( w_simu[k] * MK_train[:,:,k])
                
                predict_pr_tr.append( (clf.predict_proba(K_w))[:,clf.classes_==1])
                
            y_pred_old = np.argmax(np.array(predict_pr_tr), axis=0).ravel()
            y_pred = np.zeros(y_pred_old.shape[0])
            for lbl_idx in range(len(labels)):
                y_pred[y_pred_old==lbl_idx] = labels[lbl_idx]
            acc_train = accuracy_score(y[train_idx], y_pred) * 100
            acc_list_simu_corr_train.append( acc_train )

            del MK_train, K_y

            for idx_clf in range(labels.shape[0]):
                clf = clf_list[idx_clf]
                # MK_test = Create_MK_te(np.array(data[train_idx,])[clf.support_,:], data[test_idx,]) 
                MK_test = Create_MK_te( data[train_idx,], data[test_idx,]) 
                K_w_test = np.zeros(( n_train, n_test))
                w_simu = w_list[idx_clf]
                for k in range(num_kernels):
                    K_w_test += ( w_simu[k] * MK_test[:,:,k])
                
                predict_pr_te.append( (clf.predict_proba(K_w_test.T))[:,clf.classes_==1]) 
                            
            y_pred_old = np.argmax(np.array(predict_pr_te), axis=0).ravel()
            y_pred = np.zeros(y_pred_old.shape[0])
            for lbl_idx in range(len(labels)):
                y_pred[np.where(y_pred_old==lbl_idx)] = labels[lbl_idx]
            acc_test = accuracy_score(y[test_idx], y_pred) * 100
            # y_pred = clf_simu.predict(K_w_test.T)            
            # acc_test = accuracy_score(y[test_idx], y_pred) * 100
            acc_list_simu_corr_test.append( acc_test)
            myfile.write('lambda1: '+ str(lambda1)+'\tlambda2: '+str(lambda2)+'\tlambda3: '+str( lambda3)+'\tneig_percent: '+str(neig_percent) +'\n')
            myfile.write('test: '+ str( acc_test ) +'\ttrain: '+ str( acc_train)+'\n' )
            print('test:', acc_test,'train:', acc_train )
            # print('test: '+ str( acc_test ) +'\ttrain: '+ str( acc_train)+'\n')
            # myfile.write('test: '+ str( acc_test ) +'\ttrain: '+ str( acc_train)+ '\tkta: '+ str( KTA)+'\n' )
                
            del MK_test
            myfile.write('\n' )    
        Print_func(dataset_name)


myfile.close()