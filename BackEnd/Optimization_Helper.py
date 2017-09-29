import numpy as np
import pickle


fr=file('regressor_test.pkl','rb')
clf=pickle.load(fr)






def create_position_and_velocity_vectors(n=1,d=1,x1_range=[],x1_type='',x2_range=[],x2_type='',x3_range=[],x3_type='',
                          x4_range=[],x4_type='',x5_range=[],x5_type='',x6_range=[],x6_type=''):
    '''n= number of particles,d=number of dimensions, xi_type = 'int' or 'float' '''
    pos_vec=[]
    vel_vec=[]
    Range=[]
    Types=[]
    
    Range.append(x1_range)
    Range.append(x2_range)
    Range.append(x3_range)
    Range.append(x4_range)
    Range.append(x5_range)
    Range.append(x6_range)

    Types.append(x1_type)
    Types.append(x2_type)
    Types.append(x3_type)
    Types.append(x4_type)
    Types.append(x5_type)
    Types.append(x6_type)
     
    for i in range(0,d):
        if d>6 : 
            print 'too many dimensions. max= 6'
            break
        print Types[i]
        if Types[i]=='int': 
            pos_d=np.random.randint(Range[i][1]-Range[i][0],size=n)+Range[i][0]
            V=abs(Range[i][1]-Range[i][0])
            vel_d=np.random.randint(V-(-V),size=n)-V
        elif  Types[i]=='float':
            pos_d=np.random.rand(n)*(Range[i][1]-Range[i][0])+Range[i][0]
            V=abs(Range[i][1]-Range[i][0])
            vel_d=np.random.rand(n)*(V-(-V))-V
            for j in range(0,len(vel_d)):
                if(((vel_d[j]+pos_d[j])<Range[i][0]) or  ((vel_d[j]+pos_d[j])>Range[i][1]):vel_d[j]=0
            
        
        # elif  Types[i]=='string':pos_d=np.random.choice(Range[i],n) #no support yet
        else: 
            print 'type wrong: only int, float'
            break
        pos_vec.append(pos_d)
        vel_vec.append(vel_d)
    return np.array(pos_vec).T,np.array(vel_vec).T




def Z_Rule1(vec=[]):
  ''' define the metric we use to optimize the parameters'''
    MeanError=[]

        
    params = {'n_estimators': int(vec[0]), 'max_depth': int(vec[1]), 'min_samples_split': 2,
          'learning_rate':vec[2], 'loss': 'huber','subsample':vec[3]}
    clf = ensemble.GradientBoostingRegressor(**params)
        

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
        
    error=[]
    for iy in range(0,len(y_test)):
        y=clf.predict(X_test[iy].reshape(1,-1))
        loss=float(y_test[iy])-float(y)
        if loss<0 : error.append(loss)
    MeanError.append(abs(np.array(error).mean()))
    return MeanError,mse