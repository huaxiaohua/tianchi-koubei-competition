# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta,datetime
from math import isnan

def pre():
	fr = open('dataset/shopSellPerNum','r')
	shopSellPerNum = pickle.load(fr)
	start_day = datetime(2016,8,2)
	#print start_day.strftime('%Y-%m-%d')
	end_day = datetime(2016,10,31)

	shopSellPerNumPerDay = shopSellPerNum[shopSellPerNum['time_stamp'] == start_day]
	shopId = pd.DataFrame([i for i in range(1,2001)], columns = ['shop_id'])
	del shopSellPerNumPerDay['time_stamp']
	shopSellPerNumPerDay.columns = ['shop_id',start_day.strftime('%m-%d')] 
	shopSellPerNumPerDay = pd.merge(shopId,shopSellPerNumPerDay,how = 'left',on = 'shop_id')
	start_day = start_day + timedelta(days = 1)

	while (start_day <= end_day) :
		tmp = shopSellPerNum[shopSellPerNum['time_stamp'] == start_day]
		del tmp['time_stamp']
		tmp.columns = ['shop_id',start_day.strftime('%m-%d')]
		shopSellPerNumPerDay = pd.merge(shopSellPerNumPerDay,tmp,how = 'left',on = 'shop_id')
		start_day = start_day + timedelta(days = 1)

def deleteData(preData, preLabel, ratio):
	chooseIndex = ((preData['1_open_ratio'] > ratio) & (preData['2_open_ratio'] > ratio) & (preData['3_open_ratio'] > ratio))
	
	preData = preData[chooseIndex].reset_index()
	del preData['index']

	preLabel = preLabel[chooseIndex].reset_index()
	del preLabel['index']

	return preData, preLabel

def deletePredData(preData, ratio):
	chooseIndex = (preData['1_open_ratio'] > ratio) & (preData['2_open_ratio'] > ratio) & (preData['3_open_ratio'] > ratio) \
			& (preData['23_open_ratio'] > ratio) & (preData['123_open_ratio'] > ratio) & (preData['weekday1_open_ratio'] > ratio) \
			& (preData['weekday2_open_ratio'] > ratio) & (preData['weekday3_open_ratio'] > ratio) & (preData['weekday4_open_ratio'] > ratio) \
			& (preData['weekday5_open_ratio'] > ratio) & (preData['weekday6_open_ratio'] > ratio) & (preData['weekday7_open_ratio'] > ratio)
	
	preData = preData[chooseIndex].reset_index()
	del preData['index']

	return preData


def drawData(data):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	print data.values
	ax.set_xticks([i for i in range(data.shape[1])])
	#ax.set_xticklabels(data.columns)
	ax.grid()
	ax.plot(data.values[0])
	plt.show()

def completeData(week):
	'''
	fr = open('shopSellPerNum','r')
	shopSellPerNum = pickle.load(fr)
	start_day = datetime(2016,6,7)
	#print start_day.strftime('%Y-%m-%d')
	end_day = datetime(2016,9,5)

	shopSellPerNumPerDay = shopSellPerNum[shopSellPerNum['time_stamp'] == start_day]
	shopId = pd.DataFrame([i for i in range(1,2001)], columns = ['shop_id'])
	del shopSellPerNumPerDay['time_stamp']
	shopSellPerNumPerDay.columns = ['shop_id',start_day.strftime('%m-%d')] 
	shopSellPerNumPerDay = pd.merge(shopId,shopSellPerNumPerDay,how = 'left',on = 'shop_id')
	start_day = start_day + timedelta(days = 1)

	while (start_day <= end_day) :
		tmp = shopSellPerNum[shopSellPerNum['time_stamp'] == start_day]
		del tmp['time_stamp']
		tmp.columns = ['shop_id',start_day.strftime('%m-%d')]
		shopSellPerNumPerDay = pd.merge(shopSellPerNumPerDay,tmp,how = 'left',on = 'shop_id')
		start_day = start_day + timedelta(days = 1)

	#print shopSellPerNumPerDay.shape
	shopId = [i for i in range(1,2001)]
	completeData = []
	for shop_id in shopId:
		shopPerDay = shopSellPerNumPerDay[shopSellPerNumPerDay['shop_id'] == shop_id]
		del shopPerDay['shop_id']
		week1 = shopPerDay.iloc[:,:7]     
		week2 = shopPerDay.iloc[:,7:14]   
		week3 = shopPerDay.iloc[:,14:21]  
		week4 = shopPerDay.iloc[:,21:28]  
		week5 = shopPerDay.iloc[:,28:35]  
		week6 = shopPerDay.iloc[:,35:42]  
		week7 = shopPerDay.iloc[:,42:49]  
		week8 = shopPerDay.iloc[:,49:56]  
		week9 = shopPerDay.iloc[:,56:63] 
		week10 = shopPerDay.iloc[:,63:70] 
		week11 = shopPerDay.iloc[:,70:77] 
		week12 = shopPerDay.iloc[:,77:84] 
		week13 = shopPerDay.iloc[:,84:]

		week1.columns = [i for i in range(7)]
		week2.columns = [i for i in range(7)]
		week3.columns = [i for i in range(7)]
		week4.columns = [i for i in range(7)]
		week5.columns = [i for i in range(7)]
		week6.columns = [i for i in range(7)]
		week7.columns = [i for i in range(7)]
		week8.columns = [i for i in range(7)]
		week9.columns = [i for i in range(7)]
		week10.columns = [i for i in range(7)]
		week11.columns = [i for i in range(7)]
		week12.columns = [i for i in range(7)]
		week13.columns = [i for i in range(7)]

		weekTotal = pd.concat([week1,week2,week3,week4,week5,week6,week7,week8,week9,week10,week11,week12,week13], axis = 0)
		completeDataShop = list(weekTotal.mean(axis = 0)) #得到周二～周一 历史3个月的中位数
		completeData.append(completeDataShop)
 	completeData[500] = [142,127,157,105,82,114,135]
	fw = open('dataset/completeData_mean','w')
	pickle.dump(completeData,fw)
	'''
	#print len(completeData), len(completeData[0])
	# completeData = pd.DataFrame(completeData, columns = [i for i in range(7)])
	# print completeData.dropna().shape
	# completeData = completeData.T
	# completeData = completeData.fillna(completeData.mean()).T
	# print completeData
	# print completeData[completeData.isnull().values == True]
	
	
	fr = open('dataset/completeData_median','r')
	#fr = open('completeData_mean','r')
	completeData = pickle.load(fr)
	week = week.values
	for i in range(len(week)):
		numTotal = sum(week[i])
		for j in range(len(week[0])):
			if isnan(week[i][j]) :
				# print "none is :",
				# print week[i]
				week[i][j] = completeData[i][j]
	week = pd.DataFrame(week)
	week.columns = [i for i in range(7)]
	return week
	
	
def judgeOutLier(week1,week2,week3):
	week123 = pd.concat([week1,week2,week3], axis = 1)
	week123 = week123.values
	fr = open('dataset/completeData_median','r')
	#fr = open('completeData_mean','r')
	completeData = pickle.load(fr)

	for i in range(len(week123)):
		weekTotal1 = sum(week123[i][:7])
		weekTotal2 = sum(week123[i][7:14])
		weekTotal3 = sum(week123[i][14:])
		weekTotalList = [weekTotal1,weekTotal2,weekTotal3]

		dayTotal1 = week123[i][0] + week123[i][7] + week123[i][14]
		dayTotal2 = week123[i][1] + week123[i][8] + week123[i][15]
		dayTotal3 = week123[i][2] + week123[i][9] + week123[i][16]
		dayTotal4 = week123[i][3] + week123[i][10] + week123[i][17]
		dayTotal5 = week123[i][4] + week123[i][11] + week123[i][18]
		dayTotal6 = week123[i][5] + week123[i][12] + week123[i][19]
		dayTotal7 = week123[i][6] + week123[i][13] + week123[i][20]

		dayTotalList = [dayTotal1,dayTotal2,dayTotal3,dayTotal4,dayTotal5,dayTotal6,dayTotal7]

		for j in range(len(week123[0])):
			sumTotal = weekTotalList[j / 7] #取星期
			dayTotal = dayTotalList[j % 7]	#取周几
			if ( ((float(week123[i][j]) / sumTotal >= 0.4) and (float(week123[i][j]) / dayTotal >= 0.6)) \
				or ((float(week123[i][j]) / sumTotal <= 0.03) and (float(week123[i][j]) / dayTotal <= 0.1))) :
				# print "outlier is :",
				# print week123[i],week123[i][j],i+1
				week123[i][j] = completeData[i][j % 7]

	week123 = pd.DataFrame(week123)
	week123.columns = [i for i in range(21)]
	week1 = week123.iloc[:,:7]
	week2 = week123.iloc[:,7:14]
	week3 = week123.iloc[:,14:]

	return week1,week2,week3







if __name__ == '__main__':
	#pre()
	#completeData()
	



