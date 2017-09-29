import numpy as np
import pandas as pd
import csv

def Create_product_DataFrame():
	'''Turns the product electronics .json data into a pandas dataframe'''

	i=0
	Categories0=[]
	Categories1=[]
	Categories2=[]
	Categories3=[]
	ASINCats=[]
	Price=[]

#N=20
	with open('meta_Electronics.json') as json_data:
		for row in json_data:
			i=i+1
        #if i==N : break
			data = eval(row)
			length = len(data['categories'][0])
			Categories0.append(data['categories'][0][0])
			if length>1 : Categories1.append(data['categories'][0][1])
			else : Categories1.append('NA')
			if length>2 : Categories2.append(data['categories'][0][2])
			else : Categories2.append('NA')
			if length>3 : Categories3.append(data['categories'][0][3])
			else : Categories3.append('NA')
			ASINCats.append(data['asin'])
			try:
				data['price']
			except KeyError:
				price='NA'
			else:
				price=data['price']
			Price.append(price)
			if i%50000==0 : print (i)
	print "Done: Total Products ", i

	df = pd.DataFrame({'Price':Price,'Product':ASINCats,'Category0':Categories0,'Category1':Categories1,'Category2':Categories2,'Category3':Categories3})
	return df


def Filter_to_Headphones(df=[]):
	ReviewsH=[]
	ASINH=[]
	StarsH=[]
	Cat0H=[]
	Cat1H=[]
	Cat2H=[]
	Cat3H=[]
	PriceH=[]


	dfDVD = df[df['Category3']=='DVD Players']
	dfHeadphones = df[df['Category3']=='Headphones']

	i=0
#N=5001

	with open('reviews_Electronics_5.json') as json_data:
	start_time = time.time()
	for row in json_data:
        #print i
		i=i+1
       # if i==N : break
		if i%2000==0 : print (i),'=',float(i)*100/1689188,'%'
		data = json.loads(row)
		ProductN=data['asin']
		if(len(dfHeadphones[dfHeadphones['Product']==ProductN])) : 
            #print i
			ASINH.append(ProductN)
			ReviewsH.append(data['reviewText'])
			StarsH.append(data['overall'])
			Cat0H.append(dfHeadphones[dfHeadphones['Product']==ProductN].values[0][0])
			Cat1H.append(dfHeadphones[dfHeadphones['Product']==ProductN].values[0][1])
			Cat2H.append(dfHeadphones[dfHeadphones['Product']==ProductN].values[0][2])
			Cat3H.append(dfHeadphones[dfHeadphones['Product']==ProductN].values[0][3])
			PriceH.append(dfHeadphones[dfHeadphones['Product']==ProductN].values[0][4])

	dfHeadphonesRR = pd.DataFrame({'Price':PriceH,'Product':ASINH,'Category0':Cat0H,'Category1':Cat1H,'Category2':Cat2H,'Category3':Cat3H,'Stars':StarsH,'Review':ReviewsH})

	dfHeadphonesRR.to_csv("headphonesRR_test.csv") 

	return dfHeadphonesRR












def Collect_Topics(doc_topic=[]):
	GoodReviews = doc_topic[:,0]
	BadDurability = doc_topic[:,2]
	BadCableCord = doc_topic[:,3]
	GoodCableCord = doc_topic[:,4]
	GoodHandsFree = doc_topic[:,6]
	BadReviews = doc_topic[:,7]
	GoodLevels = doc_topic[:,8]
	GoodBrand = doc_topic[:,11]+doc_topic[:,14]+doc_topic[:,26]
	GoodComfort = doc_topic[:,12]
	GoodDurability = doc_topic[:,15]
	BadService = doc_topic[:,16]
	BadHeadsetWireless = doc_topic[:,17]
	GoodSound = doc_topic[:,18]
	BadComfort = doc_topic[:,19]+doc_topic[:,21]
	GoodCase = doc_topic[:,22]
	BadValue = doc_topic[:,23]
	GoodValue = doc_topic[:,24]+doc_topic[:,25]
	GoodMic = doc_topic[:,27]
	BadLevels = doc_topic[:,29]

	BadSum=np.add(BadDurability,BadCableCord)
	BadSum=np.add(BadSum,BadReviews)
	BadSum=np.add(BadSum,BadService)
	BadSum=np.add(BadSum,BadHeadsetWireless)
	BadSum=np.add(BadSum,BadComfort)
	BadSum=np.add(BadSum,BadValue)
	BadSum=np.add(BadSum,BadLevels)

	GoodSum=np.add(GoodReviews,GoodCableCord)
	GoodSum=np.add(GoodSum,GoodHandsFree)
	GoodSum=np.add(GoodSum,GoodLevels)
	GoodSum=np.add(GoodSum,GoodBrand)
	GoodSum=np.add(GoodSum,GoodComfort)
	GoodSum=np.add(GoodSum,GoodDurability)
	GoodSum=np.add(GoodSum,GoodSound)
	GoodSum=np.add(GoodSum,GoodCase)
	GoodSum=np.add(GoodSum,GoodValue)
	GoodSum=np.add(GoodSum,GoodMic)

	FinalDF = pd.DataFrame({'Stars/Review':dfHeadphones['Stars'].values[0:50000],
                                  'Stars/Product':dfHeadphones['Product Stars'].values[0:50000],
                                  'Price':dfHeadphones['Price'].values[0:50000],
                                  'Product':dfHeadphones['Product'].values[0:50000],
                                  'Review':dfHeadphones['Review'].values[0:50000],
                                  'Bad Durability':BadDurability,
                                    'Bad CableCord':BadCableCord,
                        'Bad Reviews':BadReviews,'Bad Service':BadService,'Bad Handsfree':BadHeadsetWireless,
                        'Bad Comfort':BadComfort,'Bad Value':BadValue,'Bad Levels':BadLevels,'Bad Sum':BadSum,
                        'Good Reviews':GoodReviews,'Good CableCord':GoodCableCord,'Good Handsfree':GoodHandsFree,'Good Levels':GoodLevels,
                        'Good Brand':GoodBrand,'Good Comfort':GoodComfort,'Good Durability':GoodDurability,
                        'Good Sound':GoodSound,'Good Case':GoodCase,
                        'Good Value': GoodValue,'Good Mic':GoodMic,'Good Sum':GoodSum})

	return FinalDF

def ReviewDF_to_ProductDF(FinalDF=[]):


	dfProd=np.unique((FinalDF[('Product')].values))


	PBadDurability=[]
	PBadCableCord=[]
	PBadReviews=[]
	PBadService=[]
	PBadHeadsetWireless=[]
	PBadComfort=[]
	PBadValue=[]
	PBadLevels=[]
    
	PGoodReviews=[]
	PGoodCableCord=[]
	PGoodHandsFree=[]
	PGoodLevels=[]
	PGoodBrand=[]
	PGoodComfort=[]
	PGoodDurability=[]
	PGoodSound=[]
	PGoodCase=[]
	PGoodValues=[]
	PGoodMic=[]

	PStars=[]
	PPrice=[]
    

	i=0
	print len(dfProd)
	for hprod in dfProd :
		if i%50==0 : print i
		i=i+1
		Pro=FinalDF[FinalDF['Product']==hprod]   
		PBadDurability.append(Pro['Bad Durability'].mean())
		PBadCableCord.append(Pro['Bad CableCord'].mean())
		PBadReviews.append(Pro['Bad Reviews'].mean())
		PBadService.append(Pro['Bad Service'].mean())
		PBadHeadsetWireless.append(Pro['Bad Handsfree'].mean())
		PBadComfort.append(Pro['Bad Comfort'].mean())
		PBadValue.append(Pro['Bad Value'].mean())
		PBadLevels.append(Pro['Bad Levels'].mean())
    
		PGoodReviews.append(Pro['Good Reviews'].mean())
		PGoodCableCord.append(Pro['Good CableCord'].mean())
		PGoodHandsFree.append(Pro['Good Handsfree'].mean())
		PGoodLevels.append(Pro['Good Levels'].mean())
		PGoodBrand.append(Pro['Good Brand'].mean())
		PGoodComfort.append(Pro['Good Comfort'].mean())
		PGoodDurability.append(Pro['Good Durability'].mean())
		PGoodSound.append(Pro['Good Sound'].mean())
		PGoodCase.append(Pro['Good Case'].mean())
		PGoodValues.append(Pro['Good Value'].mean())
		PGoodMic.append(Pro['Good Mic'].mean())
    
    
    
		PStars.append(Pro['Stars/Product'].mean())
		PPrice.append(Pro['Price'].mean())


	proDF=pd.DataFrame({'Product':dfProd,'Stars':PStars,'Price':PPrice,
                   'Bad Durability':PBadDurability,
                   'Bad CableCord':PBadCableCord,
                   'Bad Reviews':PBadReviews,'Bad Service':PBadService,'Bad Handsfree':PBadHeadsetWireless,
                   'Bad Comfort':PBadComfort,'Bad Value':PBadValue,'Bad Levels':PBadLevels,
                        'Good Reviews':PGoodReviews,'Good CableCord':PGoodCableCord,'Good Handsfree':PGoodHandsFree,'Good Levels':PGoodLevels,
                        'Good Brand':PGoodBrand,'Good Comfort':PGoodComfort,'Good Durability':PGoodDurability,
                        'Good Sound':PGoodSound,'Good Case':PGoodCase,
                        'Good Value': PGoodValues,'Good Mic':PGoodMic})

	proDF.to_csv('ProDF.csv')

	return proDF












