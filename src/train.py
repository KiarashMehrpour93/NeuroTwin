from pathlib import Path
import json, joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import preprocess_subject
from features import extract_features, get_labels

MODEL_DIR=Path('models'); MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR=Path('results'); RESULTS_DIR.mkdir(exist_ok=True)
subjects=range(1,11)
Xs=[]; ys=[]; groups=[]
for subject in subjects:
    print(f'Processing subject {subject}...')
    epochs=preprocess_subject(subject); X=extract_features(epochs); y=get_labels(epochs)
    Xs.append(X); ys.append(y); groups.extend([subject]*len(y))
X=np.vstack(Xs); y=np.concatenate(ys); groups=np.asarray(groups)
splitter=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
train_idx,test_idx=next(splitter.split(X,y,groups=groups))
model=Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression(max_iter=2000,class_weight='balanced'))])
model.fit(X[train_idx],y[train_idx]); pred=model.predict(X[test_idx])
accuracy=accuracy_score(y[test_idx],pred)
print(f'Accuracy: {accuracy:.4f}'); print(classification_report(y[test_idx],pred)); print('Confusion matrix:'); print(confusion_matrix(y[test_idx],pred))
joblib.dump(model,MODEL_DIR/'baseline_model.joblib')
with open(RESULTS_DIR/'metrics.json','w',encoding='utf-8') as f:
    json.dump({'accuracy':float(accuracy),'train_samples':int(len(train_idx)),'test_samples':int(len(test_idx)),'subjects':list(subjects)},f,indent=2)
