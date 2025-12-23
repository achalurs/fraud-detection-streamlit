import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# ======================================
# APP BOOT MESSAGE (PREVENTS BLACK SCREEN)
# ======================================
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.write("🚀 App booting successfully...")

# ======================================
# SESSION STATE
# ======================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================
# SIDEBAR
# ======================================
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔍 Predict Fraud"])

# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data():
    return pd.read_csv("transactions.csv")

df_default = load_data()

# ======================================
# PREPROCESS
# ======================================
def preprocess(df):
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["TX_DAY"] = df["TX_DATETIME"].dt.day_name()

    le = LabelEncoder()
    df["CUSTOMER_ID"] = le.fit_transform(df["CUSTOMER_ID"])
    df["TERMINAL_ID"] = le.fit_transform(df["TERMINAL_ID"])
    return df

# ======================================
# TRAIN MODELS (CACHED – VERY IMPORTANT)
# ======================================
@st.cache_resource
def train_models(df):
    df_p = preprocess(df)

    X = df_p[["TX_AMOUNT", "CUSTOMER_ID", "TERMINAL_ID", "TX_HOUR"]]
    y = df_p["TX_FRAUD"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42
    )

    # Lightweight RF (Cloud-safe)
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)

    return rf, lr, scaler, iso, X_test, X_test_scaled, y_test, df_p, X.columns

# ======================================
# DASHBOARD PAGE
# ======================================
if page == "📊 Dashboard":

    st.title("💳 Fraud Detection Dashboard")

    rf, lr, scaler, iso, X_test, X_test_scaled, y_test, df_p, features = train_models(df_default)

    rf_prob = rf.predict_proba(X_test)[:, 1]
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

    rf_auc = roc_auc_score(y_test, rf_prob)
    lr_auc = roc_auc_score(y_test, lr_prob)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("RF ROC-AUC", f"{rf_auc:.2f}")
    c2.metric("LR ROC-AUC", f"{lr_auc:.2f}")
    c3.metric("Best Model", "Random Forest" if rf_auc > lr_auc else "Logistic Regression")

    # ROC Curve
    st.subheader("📈 Model Comparison (ROC Curve)")
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

    fig, ax = plt.subplots()
    ax.plot(rf_fpr, rf_tpr, label=f"RF (AUC={rf_auc:.2f})")
    ax.plot(lr_fpr, lr_tpr, label=f"LR (AUC={lr_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # Fraud Heatmap
    st.subheader("🔥 Fraud Heatmap (Hour × Day)")
    heatmap_data = (
        df_p[df_p["TX_FRAUD"] == 1]
        .groupby(["TX_DAY", "TX_HOUR"])
        .size()
        .unstack(fill_value=0)
    )

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap="Reds", ax=ax2)
    st.pyplot(fig2)

    # Anomaly Heatmap
    st.subheader("🚨 Anomaly Heatmap (Hour × Day)")
    df_p["Anomaly"] = iso.predict(df_p[features])
    anomalies = df_p[df_p["Anomaly"] == -1]

    anom_heatmap = (
        anomalies.groupby(["TX_DAY", "TX_HOUR"])
        .size()
        .unstack(fill_value=0)
    )

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.heatmap(anom_heatmap, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# ======================================
# PREDICT PAGE
# ======================================
elif page == "🔍 Predict Fraud":

    st.title("🔍 Fraud Prediction & Anomaly Detection")

    rf, lr, scaler, iso, _, _, _, _, features = train_models(df_default)

    threshold = st.slider("🎚 Fraud Threshold (%)", 30, 90, 50) / 100

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", 0.0)
        customer = st.number_input("Customer ID", 0)

    with col2:
        terminal = st.number_input("Terminal ID", 0)
        hour = st.slider("Transaction Hour", 0, 23, 12)

    if st.button("🚨 Predict"):

        sample = pd.DataFrame([[amount, customer, terminal, hour]], columns=features)

        prob = rf.predict_proba(sample)[0][1]
        anomaly = iso.predict(sample)[0]

        if prob >= threshold:
            st.error(f"🚨 Fraud | Probability: {prob*100:.2f}%")
            decision = "Fraud"
        else:
            st.success(f"✅ Genuine | Probability: {prob*100:.2f}%")
            decision = "Genuine"

        if anomaly == -1:
            st.warning("⚠️ Anomalous Transaction Detected")

        st.session_state.history.append({
            "Amount": amount,
            "Customer": customer,
            "Terminal": terminal,
            "Hour": hour,
            "Fraud Probability (%)": round(prob*100, 2),
            "Decision": decision,
            "Anomaly": "Yes" if anomaly == -1 else "No"
        })

    st.subheader("🕒 Prediction History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)

        st.download_button(
            "⬇ Download History",
            hist_df.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv"
        )
