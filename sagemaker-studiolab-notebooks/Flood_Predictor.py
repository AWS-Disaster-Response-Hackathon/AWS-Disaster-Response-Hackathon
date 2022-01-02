from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.neural_network import NARXNN
from sklearn.model_selection import train_test_split
from sktime.forecasting.model_selection import temporal_train_test_split
loss = mean_squared_error
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import linear_model,preprocessing
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import os
import pickle



class Rainfall_Predictor():
    def load_rain_data(self):
        data = pd.read_csv("/home/studio-lab-user/sagemaker-studiolab-notebooks/DATA/process_data_Main.csv", sep=",",encoding='latin-1')
        data['Month_Year'] = data['Month'].astype(str) + '/'+ data['Year'].astype(str) 
        data_pred = pd.read_csv("/home/studio-lab-user/sagemaker-studiolab-notebooks/DATA/process_data_Forecast.csv", sep=",",encoding='latin-1')
        data_pred['Month_Year'] = data_pred['Month'].astype(str) + '/'+ data_pred['Year'].astype(str) 
        return data,data_pred

    def pre_processdata(self,data,data_pred):
        data_state = data.loc[(data['State'] == 'Bihar')]
        data_statepred = data_pred.loc[(data_pred['State'] == 'Bihar')]
        y = data_state['Rain']
        y_pred = data_statepred['Rain']
        x_Month_Year = data_statepred['Month_Year'] 
        return y,y_pred,x_Month_Year


    def train_test_data_build(self,y,y_pred,x_Month_Year):
        y_train, y_test = temporal_train_test_split(y, test_size=96)
        y_train = y_train.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        x_train = np.zeros_like(y_train)
        x_test = np.zeros_like(y_test)
        y_pred = y_pred.values.reshape(-1, 1)
        x_pred = np.zeros_like(y_pred)
        x_Month_Year = x_Month_Year.values.reshape(-1, 1)
        return x_train,y_train,x_test,y_test,x_pred,y_pred,x_Month_Year

    def define_model(self):
        class NARX(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(12, 20)
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)
                self.lin4 = nn.Linear(20, 20)
                self.lin5 = nn.Linear(20, 1)
                self.relu = nn.ReLU()

            def forward(self, xb):
                z = self.lin(xb)
                z = self.relu(z)
                z = self.lin2(z)
                z = self.relu(z)
                z = self.lin3(z)
                z = self.relu(z)
                z = self.lin4(z)
                z = self.relu(z)
                z = self.lin5(z)
                return z

        narx_net = NARXNN(ylag=11,
                      xlag=1,
                      loss_func='mse_loss',
                      optimizer='Adamax',
                      epochs=60,
                      verbose=False,
                      optim_params={'betas': (0.9, 0.999), 'eps': 1e-05} # optional parameters of the optimizer
                    )
        narx_net.net = NARX() 
        return narx_net,narx_net.net



    def build_model(self,x_train,y_train,x_test,y_test,narx_net,narx_net_net):
        train_dl = narx_net.data_transform(x_train, y_train)
        valid_dl = narx_net.data_transform(x_test, y_test)
        narx_net.fit(train_dl, valid_dl)

    def predict_rainfall(self,narx_net,x_pred,y_pred,x_Month_Year):
        yhat_pred = narx_net.predict(x_pred, y_pred)
        print(f'Accuracy RMSE Value : {math.sqrt(mean_squared_error(y_pred[8:18], yhat_pred[8:18]))}')
        plt.figure(figsize=(18,6))

        #print(x_Month_Year)
        plt.plot(x_Month_Year[8:18,0],y_pred[8:18], label='Actual',color='lightcoral', marker='D', markeredgecolor='black',linewidth=4)
        plt.plot(x_Month_Year[8:18,0],yhat_pred[8:18], label='Predict', color='#4b0082', marker='D', markeredgecolor='red',linewidth=4)
        plt.plot(x_Month_Year[8:18,0],yhat_pred[8:18] * 0.90,'--', label='Predict Low', color='#4b0082', marker='D', markeredgecolor='red',linewidth=1)
        plt.plot(x_Month_Year[8:18,0],yhat_pred[8:18] * 1.10,'--', label='Predict Upper', color='#4b0082', marker='D', markeredgecolor='red',linewidth=1)
        plt.legend()

        plt.savefig("/home/studio-lab-user/sagemaker-studiolab-notebooks/Charts/Model_Prediction.png")
        return yhat_pred[11:18],yhat_pred[11:18] * 0.90,yhat_pred[11:18] * 1.10 , x_Month_Year[11:18,0]


        
        
        
class Flood_Predictor():
    def load_flood_data(self):
        data_flood = pd.read_csv("/home/studio-lab-user/sagemaker-studiolab-notebooks/DATA/flood_data.csv", sep=",",encoding='latin-1')
        data_non_flood = pd.read_csv("/home/studio-lab-user/sagemaker-studiolab-notebooks/DATA/non-flood.csv", sep=",",encoding='latin-1')
        data_combine = pd.concat([data_flood, data_non_flood], ignore_index=True).sample(frac=1)[["Rain", "Flood"]]
        return data_combine
    
    def pre_processdata(self,data_combine):
        data_combine = data_combine.values
        X = data_combine[:,0].reshape(-1,1)
        y = data_combine[:, 1].reshape(-1,1)
        return X,y
        
    def train_test_data_build(self,X,y):
        train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.10, random_state = 18)
        train_features = train_features.reshape(-1,1)
        test_features = test_features.reshape(-1,1)
        train_labels = train_labels.reshape(-1,1)
        test_labels = test_labels.reshape(-1,1)
        return train_features,test_features,train_labels,test_labels
    
    
    
    def build_model(self,train_features,test_features,train_labels,test_labels):
        clf_model = RandomForestClassifier(bootstrap=True,max_depth=66, max_features='sqrt',
                                 min_samples_leaf=1, min_samples_split=2,
                                 n_estimators=452)

        clf_model.fit(train_features, train_labels)
        lab_enc = preprocessing.LabelEncoder()
        pred_labels_encoded = lab_enc.fit_transform(test_labels)
        preds = clf_model.predict(test_features)
        print("\n\n\n\n\n ***************Flood Prediction Confusion Matrix Bases on Rain***************")     
        print(confusion_matrix(test_labels, preds))
        return clf_model
        
    def save_model(self,clf_model):
        clf_model_dir = "sagemaker-studiolab-notebooks/Model/Rain_to_flood_model.sav"
        pickle.dump(clf_model, open(clf_model_dir, 'wb'))
        return clf_model_dir
        
    def load_model(self,clf_model_dir):
        clf_model = pickle.load(open(clf_model_dir, 'rb'))
        return clf_model
    
    def predict_flood(self,clf_model_dir,rain_forecast,rain_forecast_lower_bound,rain_forecast_upper_bound,x_Month_Year):
        flood_alert = {}
        clf_model = self.load_model(clf_model_dir)
        preds_flood = clf_model.predict(rain_forecast)
        preds_flood_upper_bound = clf_model.predict(rain_forecast_upper_bound)
        preds_flood_lower_bound = clf_model.predict(rain_forecast_lower_bound)
        print("\n\n\n\n\n ***************Send Alert for flood***************")

        for (Month_Year,lbound,pred,ubound) in zip(x_Month_Year.tolist(), preds_flood_lower_bound.tolist(),                                                                                     preds_flood.tolist(),preds_flood_upper_bound.tolist()):
                flood_risk = round([lbound, pred,ubound].count(1)/3 * 100,2)
                flood_alert[Month_Year] = flood_risk
                if flood_risk > 70:
                    print(f'Red Zone : There are {flood_risk} chance the region will be hit by flood during month of {Month_Year}')
                elif flood_risk > 50:
                    print(f'Amber Zone : There are {flood_risk} chance the region will be hit by flood during month of {Month_Year}')
                
def main_rainfall(rainfall_model):
    data,data_pred = rainfall_model.load_rain_data()
    y,y_pred,x_Month_Year = rainfall_model.pre_processdata(data,data_pred)
    x_train,y_train,x_test,y_test,x_pred,y_pred,x_Month_Year = rainfall_model.train_test_data_build(y,y_pred,x_Month_Year)
    narx_net,narx_net_net = rainfall_model.define_model()
    rainfall_model.build_model(x_train,y_train,x_test,y_test,narx_net,narx_net_net)
    rain_forecast,rain_forecast_lower_bound,rain_forecast_upper_bound,x_Month_Year = rainfall_model.predict_rainfall(narx_net,x_pred,y_pred,x_Month_Year)
    return rain_forecast,rain_forecast_lower_bound,rain_forecast_upper_bound,x_Month_Year

def main_flood(flood_model,rain_forecast,rain_forecast_lower_bound,rain_forecast_upper_bound,x_Month_Year):
    data_combine = flood_model.load_flood_data()
    X,y = flood_model.pre_processdata(data_combine)
    train_features,test_features,train_labels,test_labels = flood_model.train_test_data_build(X,y)
    clf_model = flood_model.build_model(train_features,test_features,train_labels,test_labels)
    clf_model_dir = flood_model.save_model(clf_model)
    flood_model.predict_flood(clf_model_dir,rain_forecast,rain_forecast_lower_bound,rain_forecast_upper_bound,x_Month_Year)

    
def main():
    rainfall_model = Rainfall_Predictor()
    flood_model = Flood_Predictor()
    rain_forecast,rain_forecast_lower_bound,rain_forecast_upper_bound,x_Month_Year = main_rainfall(rainfall_model)
    main_flood(flood_model,rain_forecast,rain_forecast_lower_bound,rain_forecast_upper_bound,x_Month_Year)
    
if __name__=="__main__":
    main()