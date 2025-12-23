import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# ======================================
# SESSION STATE
# ======================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================
# SIDEBAR NAVIGATION
# ======================================
st.sidebar.title("📌 Menu")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔍 Predict Fraud"]
)

# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data():
    return pd.read_csv("transactions.csv")

df_default = load_data()

# ======================================
# PREPROCESS FUNCTION
# ======================================
def preprocess(data):
    data = data.copy()
    data["TX_DATETIME"] = pd.to_datetime(data["TX_DATETIME"])
    data["TX_HOUR"] = data["TX_DATETIME"].dt.hour
    data["TX_DAY"] = data["TX_DATETIME"].dt.day_name()

    le = LabelEncoder()
    data["CUSTOMER_ID"] = le.fit_transform(data["CUSTOMER_ID"])
    data["TERMINAL_ID"] = le.fit_transform(data["TERMINAL_ID"])
    return data

# ======================================
# TRAIN MODELS
# ======================================
def train_models(df):
    df_p = preprocess(df)

    X = df_p[["TX_AMOUNT", "CUSTOMER_ID", "TERMINAL_ID", "TX_HOUR"]]
    y = df_p["TX_FRAUD"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42
    )

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    return (
        rf, lr, scaler,
        X_test, X_test_scaled, y_test,
        X_res, X.columns, df_p
    )

# ======================================
# DASHBOARD PAGE
# ======================================
if page == "📊 Dashboard":

    st.title("💳 Fraud Detection Dashboard")

    (
        rf, lr, scaler,
        X_test, X_test_scaled, y_test,
        X_res, feature_names, df_p
    ) = train_models(df_default)

    # ---------------- MODEL COMPARISON ----------------
    rf_prob = rf.predict_proba(X_test)[:, 1]
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

    rf_auc = roc_auc_score(y_test, rf_prob)
    lr_auc = roc_auc_score(y_test, lr_prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("RF ROC-AUC", f"{rf_auc:.2f}")
    c2.metric("LR ROC-AUC", f"{lr_auc:.2f}")
    c3.metric("Best Model", "Random Forest" if rf_auc > lr_auc else "Logistic Regression")

    st.markdown("---")

    # ---------------- ROC CURVE ----------------
    st.subheader("📈 ROC Curve Comparison")

    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

    fig, ax = plt.subplots()
    ax.plot(rf_fpr, rf_tpr, label=f"RF (AUC={rf_auc:.2f})")
    ax.plot(lr_fpr, lr_tpr, label=f"LR (AUC={lr_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # ======================================
    # SHAP COMPARISON: RF vs LR
    # ======================================
    st.markdown("---")
    st.subheader("🧠 SHAP Feature Importance Comparison")

    # RF SHAP
    rf_explainer = shap.Explainer(rf, X_res)
    rf_shap = rf_explainer(X_res)

    fig1 = plt.figure()
    shap.plots.bar(rf_shap[..., 1], show=False)
    plt.title("Random Forest - SHAP Importance")
    st.pyplot(fig1)

    # LR SHAP (linear explainer)
    lr_explainer = shap.LinearExplainer(lr, scaler.transform(X_res))
    lr_shap = lr_explainer(scaler.transform(X_res))

    fig2 = plt.figure()
    shap.plots.bar(lr_shap, show=False)
    plt.title("Logistic Regression - SHAP Importance")
    st.pyplot(fig2)

    # ======================================
    # ANOMALY HEATMAP
    # ======================================
    st.markdown("---")
    st.subheader("🚨 Anomaly Heatmap (Hour × Day)")

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_res)

    df_p["Anomaly"] = iso.predict(df_p[feature_names])
    df_anom = df_p[df_p["Anomaly"] == -1]

    heatmap_data = (
        df_anom
        .groupby(["TX_DAY", "TX_HOUR"])
        .size()
        .unstack(fill_value=0)
    )

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap="coolwarm", ax=ax3)
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Day of Week")
    st.pyplot(fig3)

# ======================================
# PREDICT FRAUD PAGE
# ======================================
elif page == "🔍 Predict Fraud":

    st.title("🔍 Fraud Detection & Anomaly Mode")

    uploaded_file = st.file_uploader("📂 Upload Fraud Dataset (CSV)", type=["csv"])
    df = pd.read_csv(uploaded_file) if uploaded_file else df_default.copy()

    df_p = preprocess(df)

    X = df_p[["TX_AMOUNT", "CUSTOMER_ID", "TERMINAL_ID", "TX_HOUR"]]
    y = df_p["TX_FRAUD"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_res, y_res)

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)

    threshold = st.slider("🎚 Fraud Probability Threshold (%)", 30, 90, 50) / 100

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", 0.0)
        customer = st.number_input("Customer ID", 0)

    with col2:
        terminal = st.number_input("Terminal ID", 0)
        hour = st.slider("Transaction Hour", 0, 23, 12)

    if st.button("🚨 Predict"):

        sample = pd.DataFrame(
            [[amount, customer, terminal, hour]],
            columns=X.columns
        )

        prob = rf.predict_proba(sample)[0][1]
        decision = "Fraud" if prob >= threshold else "Genuine"
        anomaly = iso.predict(sample)[0]

        if decision == "Fraud":
            st.error(f"🚨 Fraud | Probability: {prob*100:.2f}%")
        else:
            st.success(f"✅ Genuine | Probability: {prob*100:.2f}%")

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

    st.markdown("---")
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
