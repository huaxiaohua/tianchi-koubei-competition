# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator
import random
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import ExtraTreesRegressor as ET
from sklearn.ensemble import RandomForestRegressor as RF
from xgboost.sklearn import XGBClassifier as XGB
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold 
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.grid_search import ParameterGrid
from sklearn.model_selection import fit_grid_point


from make_feature import preFeature	   #三周的滑动窗口
from preShop_View import addShopView, preShopInfo

	

def trainTest():
	
	train_x, train_y, test_x, test_y, pred_x = preFeature()

	numTotal1 = [500,700,1000]
	numTotal2 = [25,30]
	numTotal3 = [0.7,0.8]
	for num1 in numTotal1:
		for num2 in numTotal2:
			for num3 in numTotal3:
				clf = ET(
					n_estimators = num1,
					max_depth = num2, 
					min_samples_split = 2,
					max_features = num3,
					min_samples_leaf = 2,
					bootstrap = False,
					n_jobs = -1,
					random_state = 1,
					criterion = 'mse'
					)
				clf.fit(train_x,train_y)
				pred = (clf.predict(test_x)).round()
				#pred = clf.predict(test_x).astype(int)
				print num1, num2, num3, caculate_score(test_y.values,pred)
				#draw_feature_importance(train_x,clf)
	
def predict():
	train_x, train_y, test_x, test_y, pred_x = preFeature()
	
	train_x = pd.concat([train_x,test_x],axis = 0)
	train_y = pd.concat([train_y,test_y],axis = 0)

	clf = ET(
			n_estimators = 700,
			max_depth = 25, 
			min_samples_split = 2,
			max_features = 0.7,
			min_samples_leaf = 2,
			bootstrap = False,
			n_jobs = -1,
			random_state = 1,
			criterion = 'mse'
			)

	clf.fit(train_x,train_y)
	pred = (clf.predict(pred_x)).round()
	pred = pred.astype(int)
	print pred
	
	fw = open('predict.csv','w')
	shopId = 0
	for i in range(2000):
		shopId += 1
		predictList = [shopId]
		tmpList = list(pred[i])
		predictList.extend(tmpList * 2)
		predictList[11] = int(round(predictList[11] * 1.05))
		# predictList[10] = int(round(predictList[10] * 1.05))
		# predictList[12] = int(round(predictList[12] * 1.05))
		print >> fw, ",".join([str(j) for j in predictList])

	
def saveData(filename,data):
	fw = open(filename,'w')
	pickle.dump(data,fw)

def loadData(filename):
	fr = open(filename,'r')
	return pickle.load(fr)


def caculate_score(real,pred):

	pred = pd.DataFrame(pred,columns = [i for i in range(pred.shape[1])])
	real = pd.DataFrame(real,columns = [i for i in range(real.shape[1])])
	#print pred
	N = pred.shape[0]
	T = pred.shape[1]
	#print N,T
	n = 0
	t = 0 
	loss = 0
	while (t < T):
		n = 0
		while (n < N):
			c_it = round(pred.ix[n,t])  #预测值
			c_git = round(real.ix[n,t]) #真实值
			if (c_it == 0 and c_git == 0):
				loss += 1			
			else :
				loss += abs((float(c_it) - c_git) / (c_it + c_git))
			n += 1
		t += 1
	loss = loss / (N * T)
	return loss

def draw_feature_importance(train_x,clf):
    feature_names = train_x.columns
    feature_importance = clf.feature_importances_
    df = pd.DataFrame({'feature_names':feature_names,'feature_importances':feature_importance})
    df1 = df.sort(columns = 'feature_importances',ascending=False)

    df1.index = [i for i in range(len(df1))]
    df1 = df1.iloc[:20,:]
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.set_xticks([i for i in range(len(df1.feature_names))]) 
    #ax.set_xticklabels(df1.feature_names)
    for i in range(20):
    	print df1.iloc[i].feature_names
    ax.grid()
    ax.plot(df1.feature_importances,label = 'feature_importance')
    plt.show('hold')


if __name__ == '__main__':
	predict()
	#trainTest()


