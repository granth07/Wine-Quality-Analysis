import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('WineQT.csv')
print(df.head())
df.info()
df.describe().T
df.isnull().sum()
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

df.isnull().sum().sum()
df.hist(bins=20, figsize=(10, 10))
plt.show()
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


df = df.drop('total sulfur dioxide', axis=1)
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
df.replace({'white': 1, 'red': 0}, inplace=True)
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']
 
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)
 
xtrain.shape, xtest.shape
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
 
for i in range(3):
    models[i].fit(xtrain, ytrain)
 
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        ytest, models[i].predict(xtest)))
    print()
    metrics.plot_confusion_matrix(models[1], xtest, ytest)
plt.show()
print(metrics.classification_report(ytest,models[1].predict(xtest)))

from missingno import bar

_ = bar(data, figsize=(10, 5), color='#FF281B')

with pd.option_context('display.precision', 2):
    explore = data.describe().T.style.background_gradient(cmap='Reds')
explore

with pd.option_context('display.precision', 2):
    explore = data.describe().T.style.background_gradient(cmap='Reds')
explore

import plotly.graph_objects as go

heat = go.Heatmap(z=corr, x=X, y=X, xgap=1, ygap=1, colorscale=colorscale, colorbar_thickness=20, colorbar_ticklen=3,)
layout = go.Layout(title_text='Correlation Matrix', title_x=0.5,  width=600, height=600,  xaxis_showgrid=False, yaxis_showgrid=False,
                   yaxis_autorange='reversed')
fig = go.Figure(data=[heat], layout=layout)        
fig.show() 

from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=len(data.columns)//2)
for i, var in enumerate(data.columns):
    fig.add_trace(go.Box(y=data[var], name=var),row=i%2+1, col=i//2+1)

fig.update_traces(boxpoints='all', jitter=.3)
fig.update_layout(height=1000, showlegend=False)
fig.show()

from plotly.express import scatter_matrix

fig = scatter_matrix(data_frame = data, color = 'quality', height = 1200, labels=labels)
fig.show()


from plotly.figure_factory import create_distplot

fig = make_subplots(rows=4, cols=3, subplot_titles=data.columns)

for j,i in enumerate(data.columns):
    fig2 = create_distplot([data[i].values], [i])
    fig2.data[0].autobinx = True
    fig.add_trace(go.Histogram(fig2['data'][0], marker_color='#f94449'), row=j//3 + 1, col=j%3 + 1)
    fig.add_trace(go.Scatter(fig2['data'][1], marker_color='#de0a26'), row=j//3 + 1, col=j%3 + 1)

fig.update_layout(height=1200, showlegend=False, margin={"l": 0, "r": 0, "t": 20, "b": 0})
fig.show()


fig = make_subplots(rows = 1, cols = 2, specs = [[{"type": "pie"}, {"type": "bar"}]])
fig.add_trace(go.Pie(values = data.quality.value_counts(), labels = data.quality.value_counts().index, domain = dict(x=[0, 0.5]), 
                     marker = dict(colors = colors), hole = .3, name=''), row = 1, col = 1)
fig.add_trace(go.Bar(x = data.quality.value_counts().index, y = data.quality.value_counts(), name='', marker = dict(color = quality,
                     colorscale = colors)), row = 1, col = 2)
fig.update_layout(showlegend = False)
fig.show()


from plotly.express import box

fig = make_subplots(rows=4, cols=3, subplot_titles=[c for c in data.columns[:-1]])
for i,v in enumerate(data.columns[:-1]):
    for t in box(data, y=v, x="quality", color="quality").data:
        fig.add_trace(t, row=(i//3)+1, col=(i%3)+1)
fig.update_layout(height=1400, showlegend=False, margin={"l": 0, "r": 0, "t": 20, "b": 0})
fig.show()

from plotly.express import bar

skewness = data.skew().sort_values(ascending=True)
fig = bar(x=skewness, y=skewness.index, color=skewness.index, labels={'x': 'Skewness', 'y':'Descriptors'})
fig.update_layout(showlegend=False)
fig.add_vline(x=1, line_dash="dash", line_color="red")
fig.show()

from IPython.display import Markdown, display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import subplots, text
matplotlib inline

models = [ExtraTreesClassifier(n_estimators=900, random_state=1251), 
         CatBoostClassifier(silent=True, depth=7, random_state=86), 
         LGBMClassifier(random_state=999), 
         RandomForestClassifier(n_estimators=1000, bootstrap=False, class_weight="balanced", random_state=247), 
         XGBClassifier(max_depth=5, subsample=0.7, colsample_bytree=0.8, random_state=149)]
model_name = ['Extra Trees', 'Category Boost', 'Light Gradient Boost', 'Random Forest', 'Extreme Gradient Boost']

for i,j in zip(models, model_name):
    if j[0] == 'E':
        y_train = y_train2
        y_test = y_test2
    i.fit(X_train, y_train)
    y_pred = i.predict(X_test)
    cm = confusion_matrix(y_pred,y_test,labels=i.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = i.classes_)
    fig, ax = subplots(figsize=(6,6))
    ax.grid(False)
    disp.plot(cmap='Reds', ax=ax)
    text(8, 5,  j + '\n' + classification_report(y_test,y_pred, zero_division=1) + '\n' + "Accuracy percent: " + 
             str(accuracy_score(y_pred, y_test)*100)[:5], fontsize=12, fontfamily='Georgia', color='k',ha='left', va='bottom')






