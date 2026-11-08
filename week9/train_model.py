from pathlib import Path
import json
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.inspection import permutation_importance
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'house_data.csv')
if 'Price' not in df: raise ValueError('CSV must contain Price target')
df=df.drop_duplicates(subset='Property_ID' if 'Property_ID' in df else None).copy()
y=pd.to_numeric(df.Price,errors='coerce'); X=df.drop(columns=['Price']+(['Property_ID'] if 'Property_ID' in df else []))
for col in X:
 v=pd.to_numeric(X[col],errors='coerce')
 if v.notna().sum()>=.8*len(X): X[col]=v
valid=y.notna(); X=X.loc[valid]; y=y.loc[valid]
nums=X.select_dtypes(include=np.number).columns.tolist(); cats=[c for c in X if c not in nums]
prep=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cats)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
models={'Linear Regression':Pipeline([('prep',prep),('model',LinearRegression())]),'Random Forest':Pipeline([('prep',prep),('model',RandomForestRegressor(n_estimators=400,min_samples_leaf=2,random_state=42))])}
rows=[]; preds={}
for name,model in models.items():
 model.fit(Xtr,ytr); pred=model.predict(Xte); preds[name]=pred
 rows.append({'model':name,'MAE':mean_absolute_error(yte,pred),'MSE':mean_squared_error(yte,pred),'RMSE':mean_squared_error(yte,pred)**.5,'R2':r2_score(yte,pred)})
metrics=pd.DataFrame(rows).sort_values('RMSE'); best=metrics.iloc[0].model; pred=preds[best]
metrics.to_csv(OUT/'model_metrics.csv',index=False)
pd.DataFrame({'actual_price':yte.values,'predicted_price':pred,'residual':yte.values-pred},index=yte.index).sort_index().to_csv(OUT/'predictions.csv',index_label='row_id')
fig,ax=plt.subplots(figsize=(7,6));ax.scatter(yte,pred,alpha=.75,color='#2878a5');lo=min(float(yte.min()),float(pred.min()));hi=max(float(yte.max()),float(pred.max()));ax.plot([lo,hi],[lo,hi],'--',color='#e07a5f',label='Perfect prediction');ax.set(xlabel='Actual price',ylabel='Predicted price',title=f'Actual vs predicted — {best}');ax.legend();fig.tight_layout();fig.savefig(ROOT/'predictions_vs_actual.png',dpi=170);plt.close(fig)
residual=yte.values-pred;fig,ax=plt.subplots(figsize=(7,4));ax.scatter(pred,residual,alpha=.75);ax.axhline(0,color='#e07a5f',ls='--');ax.set(xlabel='Predicted price',ylabel='Residual (actual − predicted)',title='Residual check');fig.tight_layout();fig.savefig(OUT/'residuals.png',dpi=170);plt.close(fig)
imp=permutation_importance(models[best],Xte,yte,n_repeats=20,random_state=42,scoring='neg_mean_absolute_error'); importance=pd.DataFrame({'feature':X.columns,'importance':imp.importances_mean,'std':imp.importances_std}).sort_values('importance');importance.to_csv(OUT/'feature_importance.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4));ax.barh(importance.feature,importance.importance,color='#54a18a');ax.set(title='Permutation importance on test set',xlabel='Improvement in negative MAE');fig.tight_layout();fig.savefig(OUT/'feature_importance.png',dpi=170);plt.close(fig)
# simple data exploration image
fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].scatter(df.Area,df.Price,alpha=.6);axs[0].set(xlabel='Area',ylabel='Price',title='Area vs price');df.groupby('Location').Price.median().sort_values().plot.bar(ax=axs[1],title='Median price by location');axs[1].set_ylabel('Median price');fig.tight_layout();fig.savefig(OUT/'exploratory_relationships.png',dpi=170);plt.close(fig)
report=f'''# House Price Prediction — Model Evaluation Report\n\n## Project goal\nUse house attributes to estimate the listed `Price`. This is a learning exercise, not a property appraisal.\n\n## Data and preparation\n- Source: `house_data.csv`; {len(df)} usable records and {len(X.columns)} predictor columns.\n- Target: `Price`. `Property_ID` is an identifier and is excluded.\n- Predictors: {', '.join(X.columns)}. Numeric fields are median-imputed and scaled; categorical fields are mode-imputed and one-hot encoded.\n- Split: 80% training ({len(Xtr)} rows) and 20% test ({len(Xte)} rows), random_state=42. Preprocessing is learned inside each model pipeline from training data only.\n\n## Test-set results\nBest RMSE among the two models compared: **{best}**.\n\n| Model | MAE | MSE | RMSE | R² |\n|---|---:|---:|---:|---:|\n'''+''.join(f"| {r.model} | {r.MAE:,.2f} | {r.MSE:,.2f} | {r.RMSE:,.2f} | {r.R2:.3f} |\n" for r in metrics.itertuples())+'''\nMAE is the average absolute error. MSE and RMSE give more weight to large errors. R² compares the predictions with a simple test-set mean baseline. The dataset does not identify currency, so errors are in source price units.\n\n## Interpretation\nUse `predictions_vs_actual.png` to see how closely predictions follow actual values. The residual plot and feature-importance chart provide additional diagnostics. Permutation importance is a model-specific association, not proof a feature causes prices to change.\n\n## Limitations\nThere are 300 rows, no sale date, condition score, or detailed location. The random holdout may not represent future markets. No external validation was available. Check recent comparable sales and test errors by location before practical use.\n\n## Reproduce\nInstall `requirements.txt`, then run `python train_model.py`. The metrics, predictions, and diagnostic charts are regenerated. The notebook also explains each step over seven learning days.\n'''
(ROOT/'model_evaluation_report.md').write_text(report,encoding='utf8')
(ROOT/'outputs'/'results.json').write_text(json.dumps({'rows':len(df),'train_rows':len(Xtr),'test_rows':len(Xte),'features':list(X.columns),'metrics':metrics.to_dict(orient='records'),'best_by_test_rmse':best,'currency':'unspecified'},indent=2,default=str),encoding='utf8')
# notebook with a guided seven-day progression, executable standalone sections
cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s): cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)})
md('# House Price Prediction\n\nWeek 9 machine-learning project. Follow the seven days in order. The dataset is in the same folder as this notebook.')
md('## Day 1 — Understand the data\nLoad the CSV, inspect the columns, missing values, and relationships.')
code('from pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT=Path.cwd()\nif not (ROOT/"house_data.csv").exists(): ROOT=ROOT.parent\ndf=pd.read_csv(ROOT/"house_data.csv")\nprint(df.shape); display(df.head()); display(df.isna().sum()); display(df.describe(include="all").T)\n')
code('plt.scatter(df.Area,df.Price,alpha=.6); plt.xlabel("Area"); plt.ylabel("Price"); plt.title("Area vs price"); plt.show()\ndisplay(df.groupby("Location").Price.median().sort_values(ascending=False))\n')
md('## Day 2 — Prepare data and split\nThe test set is held aside. Imputation, scaling, and category encoding happen inside the model pipeline, fitted on training data only.')
code('import numpy as np\nfrom sklearn.model_selection import train_test_split\ny=pd.to_numeric(df.Price,errors="coerce")\nX=df.drop(columns=["Price"] + (["Property_ID"] if "Property_ID" in df else []))\nfor col in X:\n v=pd.to_numeric(X[col],errors="coerce")\n if v.notna().sum() >= .8*len(X): X[col]=v\nvalid=y.notna(); X=X.loc[valid]; y=y.loc[valid]\nX_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)\nprint("Train rows:",len(X_train),"Test rows:",len(X_test))\n')
md('## Day 3 — Linear regression from scratch\nFor one input x, slope = sum((x−mean(x))(y−mean(y))) / sum((x−mean(x))²), and intercept = mean(y) − slope × mean(x). This introduces the idea; the evaluated model uses scikit-learn with multiple features.')
code('x=df.Area.astype(float).to_numpy(); yy=df.Price.astype(float).to_numpy()\nslope=((x-x.mean())*(yy-yy.mean())).sum()/((x-x.mean())**2).sum()\nintercept=yy.mean()-slope*x.mean()\nprint(f"Price ≈ {intercept:,.0f} + {slope:,.0f} × Area")\n')
md('## Day 4 — Train scikit-learn models\nCompare a linear regression baseline and a random forest using the same train/test split.')
code('from sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder,StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score\nnum=X_train.select_dtypes(include="number").columns.tolist(); cat=[c for c in X_train if c not in num]\nprep=ColumnTransformer([("num",Pipeline([(\'imputer\',SimpleImputer(strategy=\'median\')),(\'scale\',StandardScaler())]),num),("cat",Pipeline([(\'imputer\',SimpleImputer(strategy=\'most_frequent\')),(\'onehot\',OneHotEncoder(handle_unknown=\'ignore\'))]),cat)])\nmodels={"Linear Regression":Pipeline([(\'prep\',prep),(\'model\',LinearRegression())]),"Random Forest":Pipeline([(\'prep\',prep),(\'model\',RandomForestRegressor(n_estimators=400,min_samples_leaf=2,random_state=42))])}\nresults={}\nfor name,model in models.items():\n model.fit(X_train,y_train); results[name]=model.predict(X_test)\n print(name,"trained")\n')
md('## Day 5 — Evaluate and visualize\nCompare at least three metrics, then inspect actual versus predicted values.')
code('for name,prediction in results.items():\n print(name,"MAE",mean_absolute_error(y_test,prediction),"MSE",mean_squared_error(y_test,prediction),"RMSE",mean_squared_error(y_test,prediction)**.5,"R2",r2_score(y_test,prediction))\nbest=min(results,key=lambda name:mean_squared_error(y_test,results[name])**.5); pred=results[best]\nplt.scatter(y_test,pred); lo=min(y_test.min(),pred.min()); hi=max(y_test.max(),pred.max()); plt.plot([lo,hi],[lo,hi],"--",color="red"); plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title("Actual vs predicted — "+best); plt.show()\n')
md('## Day 6 — Check features and model errors\nPermutation importance estimates how much test-set error changes when one feature is shuffled. It is not a causal explanation.')
code('from sklearn.inspection import permutation_importance\npi=permutation_importance(models[best],X_test,y_test,n_repeats=20,random_state=42,scoring="neg_mean_absolute_error")\ndisplay(pd.DataFrame({"feature":X.columns,"importance":pi.importances_mean}).sort_values("importance",ascending=False))\nresidual=y_test.to_numpy()-pred\nplt.scatter(pred,residual); plt.axhline(0,color="red",ls="--"); plt.xlabel("Predicted price"); plt.ylabel("Actual − predicted"); plt.title("Residual check"); plt.show()\n')
md('## Day 7 — Interpret and present\nReview `model_evaluation_report.md` and `predictions_vs_actual.png`. The dataset has 300 properties and does not provide currency or sale dates. Treat the results as a practice model, not a real-world appraisal.')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
(ROOT/'house_price_prediction.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf8')
