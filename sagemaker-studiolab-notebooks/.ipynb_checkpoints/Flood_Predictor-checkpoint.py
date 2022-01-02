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




class Rainfall():
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
        y_train, y_test = temporal_train_test_split(y, test_size=48)
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
                      epochs=80,
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
        plt.figure(figsize=(18,6))

        #print(x_Month_Year)
        plt.plot(x_Month_Year[8:18,0],y_pred[8:18], label='Actual',color='lightcoral', marker='D', markeredgecolor='black',linewidth=4)
        plt.plot(x_Month_Year[8:18,0],yhat_pred[8:18], label='Predict', color='#4b0082', marker='D', markeredgecolor='red',linewidth=4)
        plt.plot(x_Month_Year[8:18,0],yhat_pred[8:18] * 0.90, label='Predict Low', color='#4b0082', marker='D', markeredgecolor='red',linewidth=1)
        plt.plot(x_Month_Year[8:18,0],yhat_pred[8:18] * 1.10, label='Predict Upper', color='#4b0082', marker='D', markeredgecolor='red',linewidth=1)
        plt.legend()

        plt.savefig("/home/studio-lab-user/sagemaker-studiolab-notebooks/Charts/Model_Prediction.png")

        

def main():
    rainfall_model = Rainfall()
    data,data_pred = rainfall_model.load_rain_data()
    y,y_pred,x_Month_Year = rainfall_model.pre_processdata(data,data_pred)
    x_train,y_train,x_test,y_test,x_pred,y_pred,x_Month_Year = rainfall_model.train_test_data_build(y,y_pred,x_Month_Year)
    narx_net,narx_net_net = rainfall_model.define_model()
    rainfall_model.build_model(x_train,y_train,x_test,y_test,narx_net,narx_net_net)
    rainfall_model.predict_rainfall(narx_net,x_pred,y_pred,x_Month_Year)



if __name__=="__main__":
    main()