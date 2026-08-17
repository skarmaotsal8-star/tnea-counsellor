import os
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(BASE_DIR, "dataset", "tnea*.csv"))

df_final     = None
rf_model     = None
le_branch    = LabelEncoder()
le_community = LabelEncoder()

if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cutoff_cols = ['OC','BC','BCM','MBC','SC','SCA','ST']
    df = df.drop(columns=['MBCDNC','MBCV'], errors='ignore')
    for col in cutoff_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_melted = df.melt(
        id_vars=['College Code','College Name','Branch Code','Branch Name'],
        value_vars=cutoff_cols, var_name='Community', value_name='Cutoff'
    )
    df_final = df_melted.dropna(subset=['Cutoff'])

    df_train = df_final.copy()
    df_train['Branch_Encoded']    = le_branch.fit_transform(df_train['Branch Code'])
    df_train['Community_Encoded'] = le_community.fit_transform(df_train['Community'])
    X = df_train[['College Code','Branch_Encoded','Community_Encoded']]
    y = df_train['Cutoff']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    print(f"[INFO] Model trained on {len(df_final)} records from {len(files)} file(s).")
else:
    print("[WARN] No tnea*.csv files found in dataset/ folder.")


def recommend_colleges(user_cutoff, user_community, user_branch_names):
    if df_final is None or rf_model is None:
        return None, None, None, "Model or data not loaded."

    branch_codes = (
        df_final[df_final["Branch Name"].isin(user_branch_names)]["Branch Code"].unique().tolist()
        if user_branch_names else df_final["Branch Code"].unique().tolist()
    )

    try:
        comm_enc = le_community.transform([user_community])[0]
    except Exception:
        return None, None, None, "Invalid community selected."

    colleges = df_final[['College Code','College Name','Branch Code','Branch Name']].drop_duplicates()
    results  = []

    for code in branch_codes:
        try:
            branch_enc = le_branch.transform([code])[0]
        except Exception:
            continue
        temp = colleges[colleges["Branch Code"] == code].copy()
        temp["Branch_Encoded"]    = branch_enc
        temp["Community_Encoded"] = comm_enc
        temp["Predicted_Cutoff"]  = rf_model.predict(temp[['College Code','Branch_Encoded','Community_Encoded']])

        def cat(pred):
            d = user_cutoff - pred
            if d >= 3:   return "SAFE"
            if d >= -5:  return "AMBITIOUS"
            return "DREAM"

        temp["Category"] = temp["Predicted_Cutoff"].apply(cat)
        results.append(temp)

    if not results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    final = pd.concat(results).sort_values("Predicted_Cutoff", ascending=False)
    cols  = ["College Name","Branch Name","Predicted_Cutoff"]
    return (
        final[final["Category"]=="DREAM"][cols].reset_index(drop=True),
        final[final["Category"]=="AMBITIOUS"][cols].reset_index(drop=True),
        final[final["Category"]=="SAFE"][cols].reset_index(drop=True),
        None
    )
