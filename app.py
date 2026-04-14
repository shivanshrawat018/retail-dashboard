import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import hashlib
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Smart Dashboard", layout="wide")

# =====================================================
# 🔐 AUTH SYSTEM (FINAL FIXED)
# =====================================================
USER_FILE = "users.csv"
SESSION_TIMEOUT = 900

if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username", "password", "role"]).to_csv(USER_FILE, index=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    return pd.read_csv(USER_FILE)

def save_user(username, password, role="user"):
    users = load_users()
    hashed_pw = hash_password(password)

    new_user = pd.DataFrame([[username, hashed_pw, role]],
                            columns=["username", "password", "role"])

    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_FILE, index=False)

# -------------------------------
# REMEMBER ME
# -------------------------------
if "remember" not in st.session_state:
    st.session_state["remember"] = False

if st.session_state.get("remember"):
    st.session_state["logged_in"] = True

# -------------------------------
# LOGIN FUNCTION
# -------------------------------
def login():
    st.title("Secure Login System")

    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "Login"

    option = st.radio(
        "Select Option",
        ["Login", "Sign Up"],
        index=0 if st.session_state["auth_mode"] == "Login" else 1
    )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    remember = st.checkbox("Remember Me")

    # LOGIN
    if option == "Login":
        if st.button("Login"):

            with st.spinner("Verifying login..."):
                users = load_users()

                if username in users["username"].values:
                    user = users[users["username"] == username].iloc[0]

                    if user["password"] == hash_password(password):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.session_state["role"] = user["role"]
                        st.session_state["login_time"] = time.time()

                        if remember:
                            st.session_state["remember"] = True

                        st.success("Login successful")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Incorrect password")
                else:
                    st.error("User not found")

    # SIGNUP
    else:
        if st.button("Create Account"):
            users = load_users()

            if username in users["username"].values:
                st.warning("User already exists")
            else:
                save_user(username, password)
                st.success("Account created successfully")

                st.session_state["auth_mode"] = "Login"
                time.sleep(1)
                st.rerun()

# -------------------------------
# SESSION CONTROL
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state.get("logged_in"):
    if time.time() - st.session_state.get("login_time", 0) > SESSION_TIMEOUT:
        st.warning("Session expired")
        st.session_state.clear()

if not st.session_state.get("logged_in"):
    login()
    st.stop()

# Logout
st.sidebar.write(f"User: {st.session_state['username']} ({st.session_state['role']})")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# =====================================================
# LOAD MODEL (SAFE)
# =====================================================
try:
    model = joblib.load("best_sales_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except:
    st.error("Model files missing")
    st.stop()

# =====================================================
# TITLE + GUIDE
# =====================================================
st.title("Smart Retail & AI Dashboard")
st.success(f"Welcome {st.session_state['username']}")

st.markdown("---")
st.subheader("User Guide")

st.markdown("""
Retail Dashboard:
- Upload dataset or use default
- Required: Order Date, Region, Category, Segment, Sales

AI Analyzer:
- Upload any CSV
- Auto charts + insights

Features:
- KPI Cards
- Actual vs Predicted
- Error Metrics
- Download report
""")

mode = st.radio("Select Mode", ["Retail Dashboard", "AI Analyzer"])

# =====================================================
# RETAIL DASHBOARD
# =====================================================
if mode == "Retail Dashboard":

    st.sidebar.header("Filters")

    month = st.sidebar.slider("Month", 1, 12)
    year = st.sidebar.number_input("Year", 2015, 2025)

    region = st.sidebar.selectbox("Region", ["West", "East", "Central", "South"])
    category = st.sidebar.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
    segment = st.sidebar.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])

    st.sidebar.markdown("---")
    file = st.sidebar.file_uploader("Upload Retail CSV", type=["csv"])

    try:
        df = pd.read_csv(file) if file else pd.read_csv("final_cleaned_data.csv")
    except:
        st.error("Dataset not found")
        st.stop()

    required = ["Order Date", "Region", "Category", "Segment", "Sales"]
    if not all(col in df.columns for col in required):
        st.error("Invalid dataset")
        st.stop()

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Order Date"])

    df["Month"] = df["Order Date"].dt.month
    df["Year"] = df["Order Date"].dt.year

    df_model = pd.get_dummies(df)
    for col in model_columns:
        if col not in df_model.columns:
            df_model[col] = 0
    df_model = df_model[model_columns]

    df["Predicted Sales"] = model.predict(df_model)

    # Prediction
    st.sidebar.markdown("---")
    filtered_df = df[
        (df["Region"] == region) &
        (df["Category"] == category) &
        (df["Segment"] == segment) &
        (df["Month"] == month)
    ]

    if st.sidebar.button("Predict Sales"):
        if len(filtered_df) == 0:
            st.sidebar.warning("No data")
        else:
            temp = pd.get_dummies(filtered_df)
            for col in model_columns:
                if col not in temp.columns:
                    temp[col] = 0
            temp = temp[model_columns]
            preds = model.predict(temp)
            st.sidebar.success(f"{preds.mean():.2f}")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual Sales", f"{df['Sales'].sum():,.0f}")
    col2.metric("Predicted Sales", f"{df['Predicted Sales'].sum():,.0f}")
    col3.metric("Difference", f"{(df['Sales'].sum()-df['Predicted Sales'].sum()):,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    col1.line_chart(df.groupby("Month")["Predicted Sales"].sum())
    col2.bar_chart(df.groupby("Category")["Predicted Sales"].sum())

    col3, col4 = st.columns(2)
    col3.bar_chart(df.groupby("Region")["Predicted Sales"].sum())
    col4.bar_chart(df.groupby("Segment")["Predicted Sales"].sum())

    st.markdown("---")
    st.subheader("Actual vs Predicted")
    st.line_chart(df.groupby("Month")[["Sales", "Predicted Sales"]].sum())

    st.markdown("---")
    st.subheader("Model Performance")

    mae = mean_absolute_error(df["Sales"], df["Predicted Sales"])
    rmse = np.sqrt(mean_squared_error(df["Sales"], df["Predicted Sales"]))
    r2 = r2_score(df["Sales"], df["Predicted Sales"])

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R²", f"{r2:.2f}")

    st.markdown("---")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Report", csv, "report.csv", "text/csv")

# =====================================================
# AI ANALYZER
# =====================================================
else:

    st.subheader("AI Analyzer")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        df = df.drop_duplicates()
        df.columns = df.columns.str.strip()

        num = df.select_dtypes(include=["int64", "float64"]).columns
        cat = df.select_dtypes(include=["object"]).columns

        if len(num) > 0:
            st.line_chart(df[num[0]])

        if len(cat) > 0 and len(num) > 0:
            st.bar_chart(df.groupby(cat[0])[num[0]].sum())

        if len(num) > 0:
            st.write("Max:", df[num[0]].max())

        if len(cat) > 0:
            st.write("Top Category:", df[cat[0]].value_counts().idxmax())

    else:
        st.info("Upload a CSV file")

# Footer
st.markdown("---")
st.markdown("Developed by Shivansh Rawat")
