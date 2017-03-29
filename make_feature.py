# -*- coding:utf-8 -*-
import pandas as pd
import pickle
from datetime import timedelta,datetime
from sklearn.preprocessing import PolynomialFeatures
from preData import deleteData, deletePredData, completeData, judgeOutLier
from preShop_View import addShopView, preShopInfo, addWeatherData

def preFeature(): #构造训练集 测试集 三周的窗口
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


	#下面是去掉节假日
	
	shopSellPerNumPerDay.index = shopSellPerNumPerDay['shop_id']
	del shopSellPerNumPerDay['shop_id']

	week1 = shopSellPerNumPerDay.iloc[:,:7]
	week2 = shopSellPerNumPerDay.iloc[:,7:14]
	week3 = shopSellPerNumPerDay.iloc[:,14:21]
	week4 = shopSellPerNumPerDay.iloc[:,21:28]
	week5 = shopSellPerNumPerDay.iloc[:,28:35]
	week6 = shopSellPerNumPerDay.iloc[:,35:42]
	week7 = shopSellPerNumPerDay.iloc[:,49:56]
	week8 = shopSellPerNumPerDay.iloc[:,70:77]
	week9 = shopSellPerNumPerDay.iloc[:,77:84]
	week10 = shopSellPerNumPerDay.iloc[:,84:]

	train_x_1 = makeFeature(week1,week2,week3)
	train_x_2 = makeFeature(week2,week3,week4)
	train_x_3 = makeFeature(week3,week4,week5)
	train_x_4 = makeFeature(week4,week5,week6)
	train_x_5 = makeFeature(week5,week6,week7)
	train_x_6 = makeFeature(week6,week7,week8)

	test_x = makeFeature(week7,week8,week9)
	pred_x = makeFeature(week8,week9,week10)



	week1 = completeData(week1)
	week2 = completeData(week2)
	week3 = completeData(week3)
	week4 = completeData(week4)
	week5 = completeData(week5)
	week6 = completeData(week6)
	week7 = completeData(week7)
	week8 = completeData(week8)
	week9 = completeData(week9)
	week10 = completeData(week10)

	week2, week3, week4 = judgeOutLier(week2, week3, week4)
	week3, week4, week5 = judgeOutLier(week3, week4, week5)
	week4, week5, week6 = judgeOutLier(week4, week5, week6)
	week5, week6, week7 = judgeOutLier(week5, week6, week7)
	week6, week7, week8 = judgeOutLier(week6, week7, week8)
	week7, week8, week9 = judgeOutLier(week7, week8, week9)
	week8, week9, week10 = judgeOutLier(week8, week9, week10)

	train_y_1 = week4
	train_y_2 = week5
	train_y_3 = week6
	train_y_4 = week7
	train_y_5 = week8
	train_y_6 = week9
	test_y = week10
	

	train_y_1.columns = [i for i in range(7)]
	train_y_2.columns = [i for i in range(7)]
	train_y_3.columns = [i for i in range(7)]
	train_y_4.columns = [i for i in range(7)]
	train_y_5.columns = [i for i in range(7)]
	train_y_6.columns = [i for i in range(7)]
	test_y.columns = [i for i in range(7)]

	train_x = pd.concat([train_x_1,train_x_2,train_x_3,train_x_4,train_x_5,train_x_6], axis = 0)
	train_y = pd.concat([train_y_1,train_y_2,train_y_3,train_y_4,train_y_5,train_y_6], axis = 0)


	train_x = train_x.reset_index()    #为了使concat后的索引正常
	del train_x['index']
	test_x = test_x.reset_index()
	del test_x['index']

	print train_x.dropna().shape
	print train_y.dropna().shape
	print test_x.dropna().shape
	print test_y.dropna().shape
	print pred_x.dropna().shape



	train_x_view, test_x_view, pred_x_view  = addShopView()
	del train_x_view['shop_id'], test_x_view['shop_id'], pred_x_view['shop_id']

	train_x = pd.concat([train_x,train_x_view], axis = 1)
	test_x = pd.concat([test_x,test_x_view], axis = 1)
	pred_x = pd.concat([pred_x,pred_x_view], axis = 1)

	train_x_weather, test_x_weather, pred_x_weather = addWeatherData()

	train_x = pd.concat([train_x,train_x_weather], axis = 1)
	test_x = pd.concat([test_x,test_x_weather], axis = 1)
	pred_x = pd.concat([pred_x,pred_x_weather], axis = 1)

	shopInfo = preShopInfo()

	train_x = pd.merge(train_x, shopInfo, on = 'shop_id',how = 'left')
	test_x = pd.merge(test_x, shopInfo, on = 'shop_id',how = 'left')
	pred_x = pd.merge(pred_x, shopInfo, on = 'shop_id',how = 'left')

	#train_x, train_y = deleteData(train_x, train_y, 0.99)
	#test_x, test_y = deleteData(test_x, test_y, 0.99)
	# print train_x.shape  #0.5:11000
	# print len(train_x.shop_id.unique())   #0.5:1968

	tmp = train_x.iloc[:,1:22].diff(axis = 1)
	del tmp['pay_0']
	train_x = pd.concat([train_x,tmp],axis = 1)

	tmp = test_x.iloc[:,1:22].diff(axis = 1)
	del tmp['pay_0']
	test_x = pd.concat([test_x,tmp],axis = 1)


	tmp = pred_x.iloc[:,1:22].diff(axis = 1)
	del tmp['pay_0']
	pred_x = pd.concat([pred_x,tmp],axis = 1)


	return train_x, train_y, test_x, test_y, pred_x
	
	

def makeFeature(week1,week2,week3):  #提取三周的特征
	week1_open_ratio = []
	tmpLen = week1.shape[1]
	for index, row in week1.iterrows():
		week1_open_ratio.append(round(float(sum(row.notnull())) / tmpLen,3))

	week2_open_ratio = []
	tmpLen = week2.shape[1]
	for index, row in week2.iterrows():
		week2_open_ratio.append(round(float(sum(row.notnull())) / tmpLen,3))

	week3_open_ratio = []	
	tmpLen = week3.shape[1]
	for index, row in week3.iterrows():
		week3_open_ratio.append(round(float(sum(row.notnull())) / tmpLen,3))

	week12_open_ratio = map(lambda (a,b) : (a + b) / 2,zip(week1_open_ratio,week2_open_ratio))
	week23_open_ratio = map(lambda (a,b) : (a + b) / 2,zip(week2_open_ratio,week3_open_ratio))
	week123_open_ratio = map(lambda (a,b,c) : (a + b + c) / 3,zip(week1_open_ratio,week2_open_ratio,week3_open_ratio))

	week1 = completeData(week1)
	week2 = completeData(week2)
	week3 = completeData(week3)
	
	week1, week2, week3	= judgeOutLier(week1,week2,week3)

	week1['shop_id'] = [i for i in range(1,2001)]
	week1.index = week1.shop_id
	del week1['shop_id']

	week2['shop_id'] = [i for i in range(1,2001)]
	week2.index = week2.shop_id
	del week2['shop_id']

	week3['shop_id'] = [i for i in range(1,2001)]
	week3.index = week3.shop_id
	del week3['shop_id']

	week1_mean = week1.mean(axis = 1)
	week1_max = week1.max(axis = 1)
	week1_min = week1.min(axis = 1)
	week1_median = week1.median(axis = 1)
	week1_var = week1.var(axis = 1)
	week1_std = week1.std(axis = 1)
	week1_mad = week1.mad(axis = 1)
	

	week2_mean = week2.mean(axis = 1)
	week2_max = week2.max(axis = 1)
	week2_min = week2.min(axis = 1)
	week2_median = week2.median(axis = 1)
	week2_var = week2.var(axis = 1)
	week2_std = week2.std(axis = 1)
	week2_mad = week2.mad(axis = 1)

	
	week3_mean = week3.mean(axis = 1)
	week3_max = week3.max(axis = 1)
	week3_min = week3.min(axis = 1)
	week3_median = week3.median(axis = 1)
	week3_var = week3.var(axis = 1)
	week3_std = week3.std(axis = 1)
	week3_mad = week3.mad(axis = 1)


	week1 = week1.reset_index()
	week2 = week2.reset_index()
	week3 = week3.reset_index()


	week12 = pd.merge(week1,week2,how = 'left',on = 'shop_id')
	week12.index = week12['shop_id']
	del week12['shop_id']

	week12_mean = week12.mean(axis = 1)
	week12_max = week12.max(axis = 1)
	week12_min = week12.min(axis = 1)
	week12_median = week12.median(axis = 1)
	week12_var = week12.var(axis = 1)
	week12_std = week12.std(axis = 1)
	week12_mad = week12.mad(axis = 1)


	week23 = pd.merge(week2,week3,how = 'left',on = 'shop_id')
	week23.index = week23['shop_id']
	del week23['shop_id']

	week23_mean = week23.mean(axis = 1)
	week23_max = week23.max(axis = 1)
	week23_min = week23.min(axis = 1)
	week23_median = week23.median(axis = 1)
	week23_var = week23.var(axis = 1)
	week23_std = week23.std(axis = 1)
	week23_mad = week23.mad(axis = 1)


	week23 = week23.reset_index()
	week123 = pd.merge(week1,week23, how = 'left', on = 'shop_id')
	week123.index = week123['shop_id']
	del week123['shop_id']


	week123_mean = week123.mean(axis = 1)
	week123_max = week123.max(axis = 1)
	week123_min = week123.min(axis = 1)
	week123_median = week123.median(axis = 1)
	week123_var = week123.var(axis = 1)
	week123_std = week123.std(axis = 1)
	week123_mad = week123.mad(axis = 1)

	columnList = []
	for i in range(week123.shape[1]):
		columnList.append("pay_%d" % i)
	week123.columns = columnList
	weekday1 = makeWeekdayFeature(week123,0)
	weekday2 = makeWeekdayFeature(week123,1)
	weekday3 = makeWeekdayFeature(week123,2)
	weekday4 = makeWeekdayFeature(week123,3)
	weekday5 = makeWeekdayFeature(week123,4)
	weekday6 = makeWeekdayFeature(week123,5)
	weekday7 = makeWeekdayFeature(week123,6)
	
	weekday1_mean = weekday1.mean(axis = 1)
	weekday1_max = weekday1.max(axis = 1)
	weekday1_min = weekday1.min(axis = 1)
	weekday1_median = weekday1.median(axis = 1)
	weekday1_var = weekday1.var(axis = 1)
	weekday1_std = weekday1.std(axis = 1)
	weekday1_mad = weekday1.mad(axis = 1)

	weekday2_mean = weekday2.mean(axis = 1)
	weekday2_max = weekday2.max(axis = 1)
	weekday2_min = weekday2.min(axis = 1)
	weekday2_median = weekday2.median(axis = 1)
	weekday2_var = weekday2.var(axis = 1)
	weekday2_std = weekday2.std(axis = 1)
	weekday2_mad = weekday2.mad(axis = 1)

	weekday3_mean = weekday3.mean(axis = 1)
	weekday3_max = weekday3.max(axis = 1)
	weekday3_min = weekday3.min(axis = 1)
	weekday3_median = weekday3.median(axis = 1)
	weekday3_var = weekday3.var(axis = 1)
	weekday3_std = weekday3.std(axis = 1)
	weekday3_mad = weekday3.mad(axis = 1)

	weekday4_mean = weekday4.mean(axis = 1)
	weekday4_max = weekday4.max(axis = 1)
	weekday4_min = weekday4.min(axis = 1)
	weekday4_median = weekday4.median(axis = 1)
	weekday4_var = weekday4.var(axis = 1)
	weekday4_std = weekday4.std(axis = 1)
	weekday4_mad = weekday4.mad(axis = 1)

	weekday5_mean = weekday5.mean(axis = 1)
	weekday5_max = weekday5.max(axis = 1)
	weekday5_min = weekday5.min(axis = 1)
	weekday5_median = weekday5.median(axis = 1)
	weekday5_var = weekday5.var(axis = 1)
	weekday5_std = weekday5.std(axis = 1)
	weekday5_mad = weekday5.mad(axis = 1)

	weekday6_mean = weekday6.mean(axis = 1)
	weekday6_max = weekday6.max(axis = 1)
	weekday6_min = weekday6.min(axis = 1)
	weekday6_median = weekday6.median(axis = 1)
	weekday6_var = weekday6.var(axis = 1)
	weekday6_std = weekday6.std(axis = 1)
	weekday6_mad = weekday6.mad(axis = 1)

	weekday7_mean = weekday7.mean(axis = 1)
	weekday7_max = weekday7.max(axis = 1)
	weekday7_min = weekday7.min(axis = 1)
	weekday7_median = weekday7.median(axis = 1)
	weekday7_var = weekday7.var(axis = 1)
	weekday7_std = weekday7.std(axis = 1)
	weekday7_mad = weekday7.mad(axis = 1)

	train_x = week123
	# print "................"
	# print train_x
	train_x['1_mean'] = week1_mean
	train_x['1_max'] = week1_max
	train_x['1_min'] = week1_min
	train_x['1_median'] = week1_median
	train_x['1_var'] = week1_var
	train_x['1_std'] = week1_std
	train_x['1_mad'] = week1_mad
	train_x['1_open_ratio'] = week1_open_ratio

	train_x['2_mean'] = week2_mean
	train_x['2_max'] = week2_max
	train_x['2_min'] = week2_min
	train_x['2_median'] = week2_median
	train_x['2_var'] = week2_var
	train_x['2_std'] = week2_std
	train_x['2_mad'] = week2_mad
	train_x['2_open_ratio'] = week2_open_ratio

	train_x['3_mean'] = week3_mean
	train_x['3_max'] = week3_max
	train_x['3_min'] = week3_min
	train_x['3_median'] = week3_median
	train_x['3_var'] = week3_var
	train_x['3_std'] = week3_std
	train_x['3_mad'] = week3_mad
	train_x['3_open_ratio'] = week3_open_ratio

	train_x['12_mean'] = week12_mean
	train_x['12_max'] = week12_max
	train_x['12_min'] = week12_min
	train_x['12_median'] = week12_median
	train_x['12_var'] = week12_var
	train_x['12_std'] = week12_std
	train_x['12_mad'] = week12_mad
	train_x['12_open_ratio'] = week12_open_ratio

	train_x['23_mean'] = week23_mean
	train_x['23_max'] = week23_max
	train_x['23_min'] = week23_min
	train_x['23_median'] = week23_median
	train_x['23_var'] = week23_var
	train_x['23_std'] = week23_std
	train_x['23_mad'] = week23_mad
	train_x['23_open_ratio'] = week23_open_ratio

	train_x['123_mean'] = week123_mean
	train_x['123_max'] = week123_max
	train_x['123_min'] = week123_min
	train_x['123_median'] = week123_median
	train_x['123_var'] = week123_var
	train_x['123_std'] = week123_std
	train_x['123_mad'] = week123_mad
	train_x['123_open_ratio'] = week123_open_ratio

	train_x['weekday1_mean'] = weekday1_mean
	train_x['weekday1_max'] = weekday1_max
	train_x['weekday1_min'] = weekday1_min
	train_x['weekday1_median'] = weekday1_median
	train_x['weekday1_var'] = weekday1_var
	train_x['weekday1_std'] = weekday1_std
	train_x['weekday1_mad'] = weekday1_mad


	train_x['weekday2_mean'] = weekday2_mean
	train_x['weekday2_max'] = weekday2_max
	train_x['weekday2_min'] = weekday2_min
	train_x['weekday2_median'] = weekday2_median
	train_x['weekday2_var'] = weekday2_var
	train_x['weekday2_std'] = weekday2_std
	train_x['weekday2_mad'] = weekday2_mad


	train_x['weekday3_mean'] = weekday3_mean
	train_x['weekday3_max'] = weekday3_max
	train_x['weekday3_min'] = weekday3_min
	train_x['weekday3_median'] = weekday3_median
	train_x['weekday3_var'] = weekday3_var
	train_x['weekday3_std'] = weekday3_std
	train_x['weekday3_mad'] = weekday3_mad


	train_x['weekday4_mean'] = weekday4_mean
	train_x['weekday4_max'] = weekday4_max
	train_x['weekday4_min'] = weekday4_min
	train_x['weekday4_median'] = weekday4_median
	train_x['weekday4_var'] = weekday4_var
	train_x['weekday4_std'] = weekday4_std
	train_x['weekday4_mad'] = weekday4_mad


	train_x['weekday5_mean'] = weekday5_mean
	train_x['weekday5_max'] = weekday5_max
	train_x['weekday5_min'] = weekday5_min
	train_x['weekday5_median'] = weekday5_median
	train_x['weekday5_var'] = weekday5_var
	train_x['weekday5_std'] = weekday5_std
	train_x['weekday5_mad'] = weekday5_mad


	train_x['weekday6_mean'] = weekday6_mean
	train_x['weekday6_max'] = weekday6_max
	train_x['weekday6_min'] = weekday6_min
	train_x['weekday6_median'] = weekday6_median
	train_x['weekday6_var'] = weekday6_var
	train_x['weekday6_std'] = weekday6_std
	train_x['weekday6_mad'] = weekday6_mad


	train_x['weekday7_mean'] = weekday7_mean
	train_x['weekday7_max'] = weekday7_max
	train_x['weekday7_min'] = weekday7_min
	train_x['weekday7_median'] = weekday7_median
	train_x['weekday7_var'] = weekday7_var
	train_x['weekday7_std'] = weekday7_std
	train_x['weekday7_mad'] = weekday7_mad

	
	return train_x.reset_index()

def makeWeekdayFeature(week123,weekDayNum):
	dayList = [weekDayNum,weekDayNum + 7,weekDayNum + 14]
	weekday = week123[dayList]
	return weekday

		


if __name__ == '__main__':
	preFeature()
	
