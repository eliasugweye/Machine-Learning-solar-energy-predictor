import os, json, warnings, hashlib, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

app = Flask(__name__)
app.secret_key = "solarml_secret_key_2025_x9z"
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER  = "models_saved"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER,  exist_ok=True)

FEATURES = ["temperature","humidity","wind_speed","cloud_cover","pressure","visibility","irradiance"]
TARGET   = "solar_output"

def _h(p): return hashlib.sha256(p.encode()).hexdigest()
USERS = {"admin":_h("admin123"), "solar":_h("solar2025"), "demo":_h("demo")}

# ── Physics formula ────────────────────────────────────────────────────────────
def solar_physics(temp, hum, wind, cloud, press, vis, irr, noise_std=4):
    cloud_factor = 0.03 + 0.97 * (1 - cloud / 100) ** 1.6
    wind_factor  = np.where(wind <= 5,  1.0,
                   np.where(wind <= 15, 1.0 - 0.055*(wind-5),
                   np.where(wind <= 25, 0.45 - 0.04*(wind-15), 0.05)))
    vis_factor   = np.clip(0.05 + 0.95*(vis/20)**0.4, 0.05, 1.0)
    temp_factor  = 1.0 - np.clip(0.004*(temp-25), -0.05, 0.25)
    hum_factor   = 1.0 - 0.002*np.clip(hum-40, 0, 55)
    press_factor = 0.85 + 0.15*np.clip((press-960)/60, 0, 1)
    base = irr * 0.38
    out  = base * cloud_factor * wind_factor * vis_factor * temp_factor * hum_factor * press_factor
    return np.clip(out + np.random.normal(0, noise_std, size=np.shape(out)), 0, None)

# ── Tiny ANN (numpy only) ──────────────────────────────────────────────────────
class TinyANN:
    def __init__(self, hidden=(64,32), lr=0.005, epochs=80, batch=128):
        self.hidden=hidden; self.lr=lr; self.epochs=epochs; self.batch=batch
        self.weights=[]; self.biases=[]; self.y_mean_=0; self.y_std_=1
    def _relu(self,x):  return np.maximum(0,x)
    def _drelu(self,x): return (x>0).astype(float)
    def _init(self,n):
        layers=[n]+list(self.hidden)+[1]
        for i in range(len(layers)-1):
            s=np.sqrt(2.0/layers[i])
            self.weights.append(np.random.randn(layers[i],layers[i+1])*s)
            self.biases.append(np.zeros((1,layers[i+1])))
    def _fwd(self,X):
        a=[X]
        for i,(W,b) in enumerate(zip(self.weights,self.biases)):
            z=a[-1]@W+b
            a.append(self._relu(z) if i<len(self.weights)-1 else z)
        return a
    def fit(self,X,y):
        np.random.seed(42); self._init(X.shape[1])
        self.y_mean_=y.mean(); self.y_std_=y.std()+1e-8
        yn=((y-self.y_mean_)/self.y_std_).reshape(-1,1); n=len(X)
        for _ in range(self.epochs):
            idx=np.random.permutation(n)
            for s in range(0,n,self.batch):
                bi=idx[s:s+self.batch]; Xb,yb=X[bi],yn[bi]
                a=self._fwd(Xb)
                d=np.clip((a[-1]-yb)*2/max(len(bi),1),-1,1)
                for i in range(len(self.weights)-1,-1,-1):
                    dW=a[i].T@d; db=d.sum(axis=0,keepdims=True)
                    if i>0: d=np.clip((d@self.weights[i].T)*self._drelu(a[i]),-1,1)
                    self.weights[i]-=self.lr*dW; self.biases[i]-=self.lr*db
        return self
    def predict(self,X):
        return self._fwd(X)[-1].flatten()*self.y_std_+self.y_mean_
    def save(self,p):
        joblib.dump({"W":self.weights,"b":self.biases,"hidden":self.hidden,
                     "lr":self.lr,"epochs":self.epochs,"batch":self.batch,
                     "ym":self.y_mean_,"ys":self.y_std_},p)
    @classmethod
    def load(cls,p):
        d=joblib.load(p); m=cls(d["hidden"],d["lr"],d["epochs"],d["batch"])
        m.weights=d["W"]; m.biases=d["b"]
        m.y_mean_=d.get("ym",0); m.y_std_=d.get("ys",1); return m

def get_models():
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42),
        "XGBoost":       HistGradientBoostingRegressor(max_iter=120,learning_rate=0.08,max_depth=6,random_state=42),
        "LightGBM":      HistGradientBoostingRegressor(max_iter=120,learning_rate=0.08,max_depth=5,max_leaf_nodes=31,random_state=7),
        "CatBoost":      ExtraTreesRegressor(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42),
        "ANN":           None,
    }

def make_stats(df, cols):
    return {c: {"mean":round(float(df[c].mean()),3), "std":round(float(df[c].std()),3),
                "min":round(float(df[c].min()),3),  "max":round(float(df[c].max()),3)}
            for c in cols}

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("dashboard") if session.get("user") else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        d=request.get_json(); u=d.get("username","").strip(); p=d.get("password","")
        if u in USERS and USERS[u]==_h(p):
            session["user"]=u; return jsonify({"success":True})
        return jsonify({"error":"Invalid username or password"}),401
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if not session.get("user"): return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

# ── Data ──────────────────────────────────────────────────────────────────────
@app.route("/api/generate_sample", methods=["POST"])
def generate_sample():
    if not session.get("user"): return jsonify({"error":"Unauthorized"}),401
    np.random.seed(42); n=3000
    temp=np.random.uniform(-5,48,n);   hum=np.random.uniform(5,100,n)
    wind=np.random.uniform(0,35,n);    cloud=np.random.uniform(0,100,n)
    press=np.random.uniform(945,1045,n); vis=np.random.uniform(0.2,25,n)
    irr=np.random.uniform(0,1100,n)
    solar=solar_physics(temp,hum,wind,cloud,press,vis,irr,noise_std=3)
    df=pd.DataFrame({"temperature":temp.round(2),"humidity":hum.round(2),
                     "wind_speed":wind.round(2),"cloud_cover":cloud.round(2),
                     "pressure":press.round(2),"visibility":vis.round(2),
                     "irradiance":irr.round(2),"solar_output":solar.round(3)})
    df.to_csv(os.path.join(UPLOAD_FOLDER,"dataset.csv"),index=False)
    return jsonify({"success":True,"rows":n,
                    "stats":make_stats(df,FEATURES+[TARGET]),
                    "preview":df.head(6).to_dict(orient="records")})

@app.route("/api/upload", methods=["POST"])
def upload():
    if not session.get("user"): return jsonify({"error":"Unauthorized"}),401
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    f=request.files["file"]
    if not f.filename.endswith(".csv"): return jsonify({"error":"CSV only"}),400
    path=os.path.join(UPLOAD_FOLDER,"dataset.csv"); f.save(path)
    df=pd.read_csv(path)
    missing=[c for c in FEATURES+[TARGET] if c not in df.columns]
    if missing: return jsonify({"error":f"Missing columns: {', '.join(missing)}"}),400
    return jsonify({"success":True,"rows":len(df),"cols":len(df.columns),
                    "stats":make_stats(df,FEATURES+[TARGET]),
                    "preview":df[FEATURES+[TARGET]].head(6).fillna(0).round(3).to_dict(orient="records")})

# ── Train — returns raw data, charts rendered in browser ──────────────────────
@app.route("/api/train", methods=["POST"])
def train():
    if not session.get("user"): return jsonify({"error":"Unauthorized"}),401
    path=os.path.join(UPLOAD_FOLDER,"dataset.csv")
    if not os.path.exists(path): return jsonify({"error":"No dataset. Generate or upload first."}),400

    df=pd.read_csv(path).dropna()
    X=df[FEATURES].values; y=df[TARGET].values
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler(); X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te)
    joblib.dump(sc,os.path.join(MODEL_FOLDER,"scaler.pkl"))

    models=get_models(); results={}; chart_data={}
    for name,mdl in models.items():
        t0=time.time()
        if name=="ANN":
            ann=TinyANN(); ann.fit(X_tr_s,y_tr)
            yp=ann.predict(X_te_s); ann.save(os.path.join(MODEL_FOLDER,"ANN.pkl"))
        else:
            mdl.fit(X_tr_s,y_tr); yp=mdl.predict(X_te_s)
            joblib.dump(mdl,os.path.join(MODEL_FOLDER,f"{name}.pkl"))
        yp=np.clip(yp,0,None)
        results[name]={"rmse":round(float(np.sqrt(mean_squared_error(y_te,yp))),3),
                       "mae":round(float(mean_absolute_error(y_te,yp)),3),
                       "r2":round(float(r2_score(y_te,yp)),4),
                       "time":round(time.time()-t0,2)}
        # send sample of actual vs predicted for JS charts (max 200 points)
        idx=np.random.choice(len(y_te),min(200,len(y_te)),replace=False)
        chart_data[name]={"actual":y_te[idx].round(2).tolist(),
                          "predicted":yp[idx].round(2).tolist()}

    # Feature importance from Random Forest
    rf=joblib.load(os.path.join(MODEL_FOLDER,"Random Forest.pkl"))
    fi=rf.feature_importances_.round(4).tolist()

    # Correlation matrix
    corr=df[FEATURES+[TARGET]].corr().round(3).values.tolist()
    corr_labels=FEATURES+[TARGET]

    # Full actual/predicted for timeline (80 pts)
    best=max(results,key=lambda k:results[k]["r2"])

    return jsonify({
        "success":True,
        "results":results,
        "best":best,
        "n_test":len(y_te),
        "chart_data":chart_data,
        "feature_importance":fi,
        "features":FEATURES,
        "corr":corr,
        "corr_labels":corr_labels,
        "y_test_sample":y_te[:80].round(2).tolist(),
        "preds_sample":{n:np.clip(
            joblib.load(os.path.join(MODEL_FOLDER,f"{n}.pkl")).predict(sc.transform(X_te[:80]))
            if n!="ANN" else TinyANN.load(os.path.join(MODEL_FOLDER,"ANN.pkl")).predict(sc.transform(X_te[:80])),
            0,None).round(2).tolist() for n in models}
    })

# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if not session.get("user"): return jsonify({"error":"Unauthorized"}),401
    data=request.get_json(); model_name=data.get("model","Random Forest")
    sc_path=os.path.join(MODEL_FOLDER,"scaler.pkl")
    mp=os.path.join(MODEL_FOLDER,f"{model_name}.pkl")
    if not os.path.exists(sc_path) or not os.path.exists(mp):
        return jsonify({"error":"Train models first"}),400
    sc=joblib.load(sc_path)
    vals=[float(data[f]) for f in FEATURES]
    Xs=sc.transform(np.array([vals]))
    pred=float(TinyANN.load(mp).predict(Xs)[0] if model_name=="ANN" else joblib.load(mp).predict(Xs)[0])
    irr=vals[FEATURES.index("irradiance")]
    pred=float(np.clip(pred,0,irr*0.38+10))
    return jsonify({"prediction":round(pred,3),"model":model_name})

@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    if not session.get("user"): return jsonify({"error":"Unauthorized"}),401
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    sc_path=os.path.join(MODEL_FOLDER,"scaler.pkl")
    if not os.path.exists(sc_path): return jsonify({"error":"Train models first"}),400
    df=pd.read_csv(request.files["file"])
    missing=[c for c in FEATURES if c not in df.columns]
    if missing: return jsonify({"error":f"Missing: {', '.join(missing)}"}),400
    sc=joblib.load(sc_path); Xs=sc.transform(df[FEATURES].values); out={}
    for name in ["Random Forest","XGBoost","LightGBM","CatBoost","ANN"]:
        mp=os.path.join(MODEL_FOLDER,f"{name}.pkl")
        if not os.path.exists(mp): continue
        yp=TinyANN.load(mp).predict(Xs) if name=="ANN" else joblib.load(mp).predict(Xs)
        out[name]=np.clip(yp,0,None).round(3).tolist()
    df2=df[FEATURES].copy()
    for k,v in out.items(): df2[f"pred_{k}"]=v
    return jsonify({"success":True,"rows":len(df),
                    "preview":df2.head(10).round(3).to_dict(orient="records"),
                    "models":list(out.keys())})

if __name__=="__main__":
    print("\n  ☀️  SolarML → http://localhost:5000  |  demo / demo\n")
    app.run(debug=False,port=5000,host="0.0.0.0")
