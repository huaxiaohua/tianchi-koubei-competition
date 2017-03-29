# -*- coding:utf-8 -*-
import pandas as pd
import pickle
from datetime import timedelta,datetime
from sklearn.preprocessing import LabelEncoder

def preUserView():
	shopData = pd.read_csv('dataset/shop_info.txt', header = None)
	shopData.columns = ['shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name']
	userViewData = pd.read_csv('dataset/user_view.txt', header = None)
	userViewData.columns = ['user_id','shop_id','time_stamp']
	userViewData['time_stamp'] = userViewData['time_stamp'].str.split().str[0]
	userViewData['time_stamp'] = pd.to_datetime(userViewData['time_stamp'], format = '%Y-%m-%d')
	shopId = pd.DataFrame([i for i in range(1,2001)])
	shopId.columns = ['shop_id']
	shopView = pd.merge(shopId,userViewData,on = 'shop_id', how = 'left')
	shopView = shopView.groupby(['shop_id','time_stamp']).user_id.agg(['count']).reset_index()
	#shopView.to_csv('shopView.csv', header = True, index = None)
	
	start_day = datetime(2016,8,2)
	#print start_day.strftime('%Y-%m-%d')
	end_day = datetime(2016,10,31)

	shopViewPerDay = shopView[shopView['time_stamp'] == start_day]
	shopId = pd.DataFrame([i for i in range(1,2001)], columns = ['shop_id'])
	del shopViewPerDay['time_stamp']
	shopViewPerDay.columns = ['shop_id',start_day.strftime('%m-%d')] 
	shopViewPerDay = pd.merge(shopId,shopViewPerDay,how = 'left',on = 'shop_id')
	start_day = start_day + timedelta(days = 1)

	while (start_day <= end_day) :
		tmp = shopView[shopView['time_stamp'] == start_day]
		del tmp['time_stamp']
		tmp.columns = ['shop_id',start_day.strftime('%m-%d')]
		shopViewPerDay= pd.merge(shopViewPerDay,tmp,how = 'left',on = 'shop_id')
		start_day = start_day + timedelta(days = 1)
	#shopViewPerDay.to_csv('shopViewPerDay.csv',index = None,header = True)

def addShopView():
	shopViewPerDay = pd.read_csv('dataset/shopViewPerDay.csv')
	shopViewPerDay.index = shopViewPerDay['shop_id']
	del shopViewPerDay['shop_id']

	
	weekA = shopViewPerDay.iloc[:,:7]
	weekB = shopViewPerDay.iloc[:,7:14]
	weekC = shopViewPerDay.iloc[:,14:21]
	weekD = shopViewPerDay.iloc[:,21:28]
	weekE = shopViewPerDay.iloc[:,28:35]
	weekF = shopViewPerDay.iloc[:,35:42]
	weekG = shopViewPerDay.iloc[:,49:56]
	weekH = shopViewPerDay.iloc[:,70:77]
	weekI = shopViewPerDay.iloc[:,77:84]
	weekJ = shopViewPerDay.iloc[:,84:]

	train_x_1 = makeFeature(weekA,weekB,weekC)
	train_y_1 = weekD
	train_x_2 = makeFeature(weekB,weekC,weekD)
	train_y_2 = weekE
	train_x_3 = makeFeature(weekC,weekD,weekE)
	train_y_3 = weekF	
	train_x_4 = makeFeature(weekD,weekE,weekF)
	train_y_4 = weekG
	train_x_5 = makeFeature(weekE,weekF,weekG)
	train_y_5 = weekH	
	train_x_6 = makeFeature(weekF,weekG,weekH)
	train_y_6 = weekI

	test_x = makeFeature(weekG,weekH,weekI)
	
	pred_x = makeFeature(weekH,weekI,weekJ)


	train_x = pd.concat([train_x_1,train_x_2,train_x_3,train_x_4,train_x_5,train_x_6], axis = 0)
	#train_x = train_x_6

	train_x = train_x.reset_index()
	del train_x['index']

	train_x = train_x.fillna(0)
	test_x = test_x.fillna(0)
	pred_x = pred_x.fillna(0)
	#print test_x
	return train_x, test_x, pred_x



	

def makeFeature(weekA,weekB,weekC):
	weekA = weekA.reset_index()
	weekB = weekB.reset_index()
	weekC = weekC.reset_index()
	week12 = pd.merge(weekA,weekB,on = 'shop_id')
	week123 = pd.merge(week12,weekC,on = 'shop_id')
	columnList = ['shop_id']
	for i in range(week123.shape[1] - 1):
		columnList.append("view_%d" % i)
	week123.columns = columnList
	return week123



def preShopInfo():
	shopInfo = pd.read_csv('dataset/shop_info.txt', header = None)
	shopInfo.columns = ['shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name']
	
	enc = LabelEncoder()
	shopInfo['city_name'] = enc.fit_transform(shopInfo['city_name'])
	shopInfo['cate_1_name'] = enc.fit_transform(shopInfo['cate_1_name'])
	shopInfo['cate_2_name'] = enc.fit_transform(shopInfo['cate_2_name'])
	del shopInfo['cate_3_name']
	shopInfo = shopInfo.fillna(0)
	return shopInfo

	'''
	enc = LabelEncoder()
	#shopInfo['cate_2_name'] = enc.fit_transform(shopInfo['cate_2_name'])
	cate_2_name = shopInfo['cate_2_name'].unique()
	for name in cate_2_name:
		print name,shopInfo[shopInfo['cate_2_name'] == name].shape[0]
	'''
def addWeatherData():
	weatherData = pd.read_csv("dataset/weather_all.csv", header = None)
	weatherData.columns = ['city_name','time_stamp','max_temp','min_temp','weather','wind','wind_degree']
	del weatherData['wind'], weatherData['wind_degree']
	weatherData['time_stamp'] = pd.to_datetime(weatherData['time_stamp'],format = '%Y-%m-%d')
	start_day = datetime(2016,8,2)
	end_day = datetime(2016,11,7)
	weatherData = weatherData[(weatherData['time_stamp'] >= start_day) & (weatherData['time_stamp'] <= end_day)]
	#print weatherData

	shopInfo = pd.read_csv('dataset/shop_info.txt', header = None)
	shopInfo.columns = ['shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name']
	del shopInfo['location_id'], shopInfo['per_pay'], shopInfo['score'], shopInfo['comment_cnt'], shopInfo['shop_level'], shopInfo['cate_1_name'], shopInfo['cate_2_name'], shopInfo['cate_3_name']
	weather_level = pd.read_csv("dataset/weather_list.csv",header = None)
	weather_level.columns = ['weather','weather_level']
	#print weatherData
	enc = LabelEncoder()
	shopInfo['city_name'] = enc.fit_transform(shopInfo['city_name'])
	weatherData['city_name'] = enc.transform(weatherData['city_name'])
	weatherData['weather'] = enc.fit_transform(weatherData['weather'])
	weather_level['weather'] = enc.transform(weather_level['weather'])
	
	weatherMaxTempPerDay = pd.DataFrame()
	weatherMinTempPerDay = pd.DataFrame()
	weatherPerDay = pd.DataFrame()

	weatherMaxTempPerDay['city_name'] = weatherData['city_name'].unique()
	weatherMinTempPerDay['city_name'] = weatherData['city_name'].unique()
	weatherPerDay['city_name'] = weatherData['city_name'].unique()

	while (start_day <= end_day) :
		tmp = weatherData[weatherData['time_stamp'] == start_day]
		del tmp['time_stamp'], tmp['min_temp'], tmp['weather']
		columnName = "max_temp_%s" % start_day.strftime('%m-%d')
		tmp.columns = ['city_name',columnName]
		weatherMaxTempPerDay = pd.merge(weatherMaxTempPerDay,tmp,on = 'city_name')

		tmp = weatherData[weatherData['time_stamp'] == start_day]
		del tmp['time_stamp'], tmp['max_temp'], tmp['weather']
		columnName = "min_temp_%s" % start_day.strftime('%m-%d')
		tmp.columns = ['city_name',columnName]
		weatherMinTempPerDay = pd.merge(weatherMinTempPerDay,tmp,on = 'city_name')

		tmp = weatherData[weatherData['time_stamp'] == start_day]
		del tmp['time_stamp'], tmp['max_temp'], tmp['min_temp']
		tmp = pd.merge(tmp,weather_level, on = 'weather', how = 'left')
		del tmp['weather']
		columnName = "weather_level_%s" % start_day.strftime('%m-%d')
		tmp.columns = ['city_name',columnName]
		weatherPerDay = pd.merge(weatherPerDay,tmp,on = 'city_name')


		start_day = start_day + timedelta(days = 1)

	weatherMaxTempPerDay = pd.merge(weatherMaxTempPerDay, shopInfo, on = 'city_name', how = 'right')
	del weatherMaxTempPerDay['city_name']
	weatherMinTempPerDay = pd.merge(weatherMinTempPerDay, shopInfo, on = 'city_name', how = 'right')
	del weatherMinTempPerDay['city_name']
	weatherPerDay = pd.merge(weatherPerDay, shopInfo, on = 'city_name', how = 'right')
	del weatherPerDay['city_name']

	weatherMaxTempPerDay = weatherMaxTempPerDay.sort(columns = 'shop_id',ascending = True).reset_index()
	del weatherMaxTempPerDay['index']

	weatherMinTempPerDay = weatherMinTempPerDay.sort(columns = 'shop_id',ascending = True).reset_index()
	del weatherMinTempPerDay['index']

	weatherPerDay = weatherPerDay.sort(columns = 'shop_id',ascending = True).reset_index()
	del weatherPerDay['index']

	#print weatherPerDay

	train_x_max_temp, test_x_max_temp, pred_x_max_temp = mergeWeatherFeature(weatherMaxTempPerDay, "max_tmp")
	train_x_min_temp, test_x_min_temp, pred_x_min_temp = mergeWeatherFeature(weatherMinTempPerDay, "min_tmp")
	train_x_weather, test_x_weather, pred_x_weather = mergeWeatherFeature(weatherPerDay, "weather")
	
	train_x = pd.concat([train_x_max_temp,train_x_min_temp], axis = 1)
	train_x = pd.concat([train_x,train_x_weather], axis = 1)

	test_x = pd.concat([test_x_max_temp,test_x_min_temp], axis = 1)
	test_x = pd.concat([test_x,test_x_weather], axis = 1)

	pred_x = pd.concat([pred_x_max_temp,pred_x_min_temp], axis = 1)
	pred_x = pd.concat([pred_x,pred_x_weather], axis = 1)

	return train_x, test_x, pred_x



def mergeWeatherFeature(data,dataType):
	data.index = data['shop_id']
	del data['shop_id']
	week1 = data.iloc[:,:7]     #8.2 ~ 8.8
	week2 = data.iloc[:,7:14]   #8.9 ~ 8.15
	week3 = data.iloc[:,14:21]  #8.16 ~ 8.22
	week4 = data.iloc[:,21:28]  #8.23 ~ 8.29
	week5 = data.iloc[:,28:35]  #8.30 ~ 9.5

	week6 = data.iloc[:,35:42]  #9.6 ~ 9.12   教师节
	week7 = data.iloc[:,49:56]  #9.20 ~ 9.26
	week8 = data.iloc[:,70:77] #10.11 ~ 10.17
	week9 = data.iloc[:,77:84] #10.18 ~ 10.24
	week10 = data.iloc[:,84:91] #10.25 ~ 10.31
	week11 = data.iloc[:,91:]	#11.1 ~ 11.7

	train_x_1 = makeWeatherFeature(week4,dataType)
	
	train_x_2 = makeWeatherFeature(week5,dataType)
	
	train_x_3 = makeWeatherFeature(week6,dataType)
		
	train_x_4 = makeWeatherFeature(week7,dataType)

	train_x_5 = makeWeatherFeature(week8,dataType)

	train_x_6 = makeWeatherFeature(week9,dataType)

	test_x = makeWeatherFeature(week10,dataType)

	pred_x = makeWeatherFeature(week11,dataType)

	


	train_x = pd.concat([train_x_1,train_x_2,train_x_3,train_x_4,train_x_5,train_x_6], axis = 0)
	#train_x = train_x_6
	train_x = train_x.reset_index()
	del train_x['index']
	del train_x['shop_id'], test_x['shop_id'], pred_x['shop_id']
	return train_x, test_x, pred_x


def makeWeatherFeature(week,dataType):
	week = week.reset_index()
	columnList = ['shop_id']
	for i in range(week.shape[1] - 1):
		columnList.append("%s_%d" % (dataType,i))
	week.columns = columnList
	return week

def preWeather():  #将天气分类 分为好中坏三类
	weatherData = pd.read_csv("dataset/weather_all.csv", header = None)
	weatherData.columns = ['city_name','time_stamp','max_temp','min_temp','weather','wind','wind_degree']
	del weatherData['wind'], weatherData['wind_degree']
	weatherData['time_stamp'] = pd.to_datetime(weatherData['time_stamp'],format = '%Y-%m-%d')
	start_day = datetime(2016,8,2)
	end_day = datetime(2016,11,7)
	weatherData = weatherData[(weatherData['time_stamp'] >= start_day) & (weatherData['time_stamp'] <= end_day)]
	weather = weatherData.weather.unique()
	# fw = open("weather_list.csv",'w')
	# for i in weather:
	# 	print i

	# weather_level = pd.read_csv("weather_list.csv",header = None)
	# weather_level.columns = ['weather','weather_level']
	# print weather_level['weather_level']










if __name__ == '__main__':
	#preUserView()
	#addShopView()
	#preShopInfo()
	#addShopView()
	#print addWeatherData()
	#preWeather()
	addWeatherData()

