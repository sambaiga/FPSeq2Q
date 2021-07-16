import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from net.metrics import prediction_error_metrics, mean_squared_error
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os

def save_model(model, file_name):
    dump(model, file_name) 


def load_model(file_name):
    return load(file_name) 

def get_metrics(mu, true, R):
    
    true_nmpic=2*true.std()/R

    nrmse=np.sqrt(mean_squared_error(true, mu))/R       
    metrics=dict(nrmse=nrmse)
    metrics=pd.DataFrame.from_dict(metrics, orient='index').T
    pred=prediction_error_metrics(mu, true)
    MAAP = np.arctan(np.abs((true-mu)/true)).mean()
    metrics['mdae']=pred['mdae']
    metrics['mae']=pred['mae']
    metrics['corr']=pred['corr']
    metrics['r2']=pred['r2']
    metrics['marpd']=pred['marpd']
    metrics['maap']=MAAP
    
    return metrics

def make_pipeline(model, n_components=2):
    steps = list()
    steps.append(('pca', PCA(n_components=n_components)))
    steps.append(('model', model))
    pipeline = Pipeline(steps=steps)
    return pipeline

def get_prediction(X_train, X_test, pipeline, points=48):
    outputs = []

    for i in range(0, len(X_test)):
        data = X_train[-1] if i==0 else  X_test[i]
        for i in range (0, points):
            y_pred = pipeline.predict(np.array([data]))[0]
            outputs.append(y_pred)
            new_sample = data[1:]
            new_sample = np.append(new_sample, y_pred)
            data = new_sample
    pred=np.array(outputs).reshape(-1, points)
    return pred


def fit_baseline(train_loader, test_loader, file_name, baseline='RF'):
    
    # fit the model
    if baseline=='RF':
        model=RandomForestRegressor()
    else:
        model=MultiOutputRegressor(SVR(kernel='rbf', C=1, gamma='auto', epsilon=0.01))
    pipeline = make_pipeline(model)
    X_train, y_train = train_loader.dataset.tensors
 
    pipeline.fit(X_train.numpy().reshape(len(X_train), -1), y_train.numpy())
    
    X_test, y_test = test_loader.dataset.tensors
    y_hat = pipeline.predict(X_test.numpy().reshape(len(X_test), -1))

    save_model(pipeline, f'{baseline}_{file_name}.joblib')
    
    
    return y_hat, y_test
        



