import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import zipfile
import io
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Federal Republic of Nigeria & IBS - Forensic Audit Portal",
    page_icon="🇳🇬",
    layout="wide"
)

# ============================================================
# CUSTOM STYLING (Clean & Error-Free)
# ============================================================
st.markdown("""
<style>
.stApp {
    background-color: #f7f9f8;
}
.nigeria-banner {
    background-color: #008751;
    height: 10px;
    border-radius: 4px;
    margin-bottom: 15px;
}
.top-header {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border: 2px solid #008751;
    box-shadow: 0px 4px 15px rgba(0, 135, 81, 0.15);
    text-align: center;
    margin-bottom: 15px;
}
.top-title {
    color: #008751;
    font-weight: 900;
    font-size: 20px;
    text-transform: uppercase;
    margin: 5px 0 2px 0;
}
.top-sub {
    color: #1b5e20;
    font-size: 12px;
    font-weight: 700;
    margin: 0;
}
.stButton>button {
    background-color: #008751 !important;
    color: white !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHART FUNCTIONS DEFINITION (Placed at Top)
# ============================================================
def chart_pca(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    n = d[~d["flagged"]]
    f = d[d["flagged"]]
    ax.scatter(n["pc1"], n["pc2"], s=12, c="#008751", label="Normal", alpha=0.7)
    ax.scatter(f["pc1"], f["pc2"], s=35, c="#d32f2f", marker="x", label="Flagged")
    ax.set_title("Payroll Records (PCA View)", fontweight='bold', fontsize=10)
    ax.set_xlabel("Principal Component 1", fontsize=9)
    ax.set_ylabel("Principal Component 2", fontsize=9)
    ax.legend(fontsize=8)
    return fig

def chart_lga(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    if "lga" in d.columns:
        c = d[d["flagged"]]["lga"].value_counts()
        if not c.empty:
            ax.barh(c.index, c.values, color="#1b5e20")
            ax.set_title("Flagged Staff per Local Government", fontweight='bold', fontsize=10)
            ax.set_xlabel("Flagged Count", fontsize=9)
            ax.set_ylabel("LGA", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No LGA Flagged Records", ha='center', va='center')
            ax.set_title("Flagged Staff per Local Government", fontweight='bold', fontsize=10)
    else:
        ax.text(0.5, 0.5, "LGA Column Missing", ha='center', va='center')
    return fig

def chart_status_bar(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = d["anomalystatus"].value_counts().reindex(["No Anomaly Flag", "Flagged for Review"], fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=["#008751", "#d32f2f"])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}', ha='center', va='bottom', fontsize=9)
    ax.set_title("Anomaly Status Count (Bar Chart)", fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', rotation=10)
    return fig

def chart_status_pie(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    pie_counts = d["anomalystatus"].value_counts().reindex(["No Anomaly Flag", "Flagged for Review"], fill_value=0)
    colors = ["#008751", "#d32f2f"]
    ax.pie(
        pie_counts.values,
        labels=pie_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 8}
    )
    ax.set_title("Anomaly Share (%) (Pie Chart)", fontweight='bold', fontsize=10)
    return fig

def chart_salary_dist(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    if "netsalary" in d.columns:
        ax.hist(d["netsalary"], bins=15, color="#008751", edgecolor="white", alpha=0.85)
        ax.set_title("Net Salary Distribution Frequency", fontweight='bold', fontsize=10)
        ax.set_xlabel("Net Salary (₦)", fontsize=9)
        ax.set_ylabel("Employee Count", fontsize=9)
    else:
        ax.text(0.5, 0.5, "NetSalary Column Missing", ha='center', va='center')
    return fig

def chart_allowance_scatter(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    if "allowances" in d.columns and "netsalary" in d.columns:
        n = d[~d["flagged"]]
        f = d[d["flagged"]]
        ax.scatter(n["allowances"], n["netsalary"], color="#008751", label="Normal", alpha=0.6, s=25)
        ax.scatter(f["allowances"], f["netsalary"], color="#d32f2f", label="Flagged", marker="X", s=70)
        ax.set_title("Allowances vs Net Salary Outlier Mapping", fontweight='bold', fontsize=10)
        ax.set_xlabel("Allowances (₦)", fontsize=9)
        ax.set_ylabel("Net Salary (₦)", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Required Financial Columns Missing", ha='center', va='center')
    return fig

# ============================================================
# TOP HEADER: TINY EFCC AT EXTREME TOP RIGHT, COAT OF ARMS & IBS LOGO BALANCED LEFT/RIGHT
# ============================================================
col_top_spacer, col_tiny_efcc = st.columns([19, 1])
with col_tiny_efcc:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    if os.path.exists("efcc_logo.jpeg"):
        st.image("efcc_logo.jpeg", width=25)
    elif os.path.exists("efcc_logo.jpg"):
        st.image("efcc_logo.jpg", width=25)
    else:
        st.write("🦅")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

col_arm, col_title, col_school = st.columns([1, 4, 1])

with col_arm:
    st.markdown("<div style='text-align: left;'>", unsafe_allow_html=True)
    if os.path.exists("coat_of_arms.jpeg"):
        st.image("coat_of_arms.jpeg", width=80)
    elif os.path.exists("coat_of_arms.jpg"):
        st.image("coat_of_arms.jpg", width=80)
    else:
        st.write("🇳🇬 Coat of Arms")
    st.markdown("</div>", unsafe_allow_html=True)

with col_title:
    st.markdown("""
        <div class="top-header" style="margin-top: 5px; padding: 10px;">
            <p class="top-title">International Business School</p>
            <p class="top-sub">AI Payroll Anomaly Detection & EFCC Forensic Audit Portal</p>
        </div>
    """, unsafe_allow_html=True)

with col_school:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    logo_found = False
    for filename in ["school_logo.jpeg", "school_logo.jpg", "school_logo.png", "logo.jpeg", "logo.jpg", "logo.png"]:
        if os.path.exists(filename):
            st.image(filename, width=80)
            logo_found = True
            break

    if not logo_found:
        st.write("🏫 IBS Logo")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="nigeria-banner" style="margin-top: 10px;"></div>', unsafe_allow_html=True)

# ============================================================
# INITIALIZE SESSION STATES (Starts at Step 1)
# ============================================================
if "auth_verified" not in st.session_state:
    st.session_state.auth_verified = False
    st.session_state.full_name = ""
    st.session_state.address = ""
    st.session_state.contact = ""

if "step" not in st.session_state:
    st.session_state.step = 1

if "dev_index" not in st.session_state:
    st.session_state.dev_index = 0

# ============================================================
# MAIN BODY STEP NAVIGATION BAR (WITH STAGE LOCKS)
# ============================================================
st.markdown("### 📌 Navigation Menu")
step_names = {
    1: "1. Authorization",
    2: "2. Data Input",
    3: "3. AI Analysis",
    4: "4. Visualizations",
    5: "5. Q&A & Kudos",
    6: "6. Developers"
}

nav_cols = st.columns(6)
for s_num, s_name in step_names.items():
    with nav_cols[s_num - 1]:
        btn_label = f"📍 {s_num}" if st.session_state.step != s_num else f"👉 [{s_num}]"
        if st.button(btn_label, use_container_width=True, key=f"nav_btn_{s_num}"):
            if s_num > 1 and (not st.session_state.auth_verified or not st.session_state.full_name.strip()):
                st.warning("⛔ ACCESS LOCKED: Please complete and confirm valid authorization on Step 1 first.")
            elif s_num > 2 and "df" not in st.session_state:
                st.warning("⛔ ACCESS LOCKED: Please load or generate a dataset on Step 2 first.")
            elif s_num > 3 and "results" not in st.session_state:
                st.warning("⛔ ACCESS LOCKED: Please run AI Model Analysis on Step 3 first.")
            else:
                st.session_state.step = s_num
                st.rerun()

st.markdown(f"**Current Active Section:** `{step_names[st.session_state.step]}`")
st.divider()

# ============================================================
# PAGE 1: AUTHORIZATION & REGISTRATION
# ============================================================
if st.session_state.step == 1:
    st.header("🔑 Investigator Authorization & Registration")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Authorization Check")
        auth_choice = st.radio(
            "Do you have official authorization?",
            ["-- Please Select --", "YES", "NO"],
            index=0
        )

        if auth_choice == "-- Please Select --":
            st.session_state.auth_verified = False
            st.info("ℹ️ Please select your authorization status to proceed.")
        elif auth_choice == "NO":
            st.session_state.auth_verified = False
            st.error("⛔ ACCESS DENIED: Proper authorization is required.")
        else:
            st.session_state.auth_verified = True
            st.success("✅ Authorization Confirmed.")

    with col2:
        st.subheader("Investigator Details")
        st.session_state.full_name = st.text_input("Full Name:", value=st.session_state.full_name)
        st.session_state.address = st.text_input("Official Address:", value=st.session_state.address)
        st.session_state.contact = st.text_input("Contact Number:", value=st.session_state.contact)

    if st.session_state.auth_verified and st.session_state.full_name.strip() != "":
        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 5px solid #008751; margin-top: 15px;">
                <h4 style="color: #008751; margin: 0 0 5px 0;">Official Session Active</h4>
                <p style="margin: 0; color: #1b5e20; font-weight: 600;">
                    Welcome, <b>{st.session_state.full_name}</b>. Your credentials have been logged for this forensic session.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")
    if st.button("Continue to Data Input & Cleaning ➡️", use_container_width=True):
        if not st.session_state.auth_verified:
            st.warning("⚠️ Please complete valid authorization first.")
        elif not st.session_state.full_name.strip():
            st.warning("⚠️ Please enter your full name to proceed.")
        else:
            st.session_state.step = 2
            st.rerun()

# ============================================================
# PAGE 2: DATA INPUT & CLEANING
# ============================================================
if st.session_state.step == 2:
    if not st.session_state.auth_verified or not st.session_state.full_name.strip():
        st.error("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 to access this page.")
        if st.button("⬅️ Go Back to Step 1: Authorization", key="back_step2_auth"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    st.header("📂 Payroll Data Input & Cleaning")

    data_choice = st.radio(
        "Select Dataset Source:",
        ["— Please Select —", "Generate Sample Payroll Dataset (500 Records)", "Upload External Payroll CSV File"],
        horizontal=True,
        index=0,
        key="dataset_radio_choice"
    )

    df = None
    if data_choice == "— Please Select —":
        if "df" in st.session_state:
            del st.session_state.df
        st.info("ℹ️ Please select a dataset source above to proceed.")

    elif "Generate" in data_choice:
        np.random.seed(42)
        num_emp = 500
        df = pd.DataFrame({
            "employeeid": [f"EMP{i:03d}" for i in range(1, num_emp + 1)],
            "name": [f"Employee_{i:03d}" for i in range(1, num_emp + 1)],
            "state": np.random.choice(["Kogi State", "Kwara State", "Ekiti State"], num_emp),
            "lga": np.random.choice(["Okene", "Kabba", "Lokoja", "Ado-Ekiti", "Ijumu", "Yagba East"], num_emp),
            "basicsalary": np.maximum(np.random.normal(85000, 15000, num_emp), 30000).round(2),
            "allowances": np.maximum(np.random.normal(18000, 5000, num_emp), 0).round(2),
            "deductions": np.maximum(np.random.normal(6000, 1800, num_emp), 0).round(2)
        })
        df["netsalary"] = (df["basicsalary"] + df["allowances"] - df["deductions"]).round(2)
        df.loc[10, "basicsalary"] = 450000.00
        df.loc[10, "netsalary"] = df.loc[10, "basicsalary"] + df.loc[10, "allowances"] - df.loc[10, "deductions"]
        df.loc[25, "allowances"] = 300000.00
        df.loc[25, "netsalary"] = df.loc[25, "basicsalary"] + df.loc[25, "allowances"] - df.loc[25, "deductions"]
        df.loc[50, "deductions"] = 0.00
        st.success("Sample Nigerian payroll dataset generated with test anomalies.")

    elif "Upload" in data_choice:
        uploaded_file = st.file_uploader("Upload Payroll CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower().strip() for c in df.columns]
            st.success("Successfully loaded uploaded CSV file.")
        else:
            if "df" in st.session_state:
                del st.session_state.df
            st.info("Awaiting CSV upload...")

    if df is not None:
        st.session_state.df = df
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to Authorization", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_b2:
        can_proceed = "df" in st.session_state and st.session_state.df is not None
        if st.button("Continue to AI Model Analysis ➡️", use_container_width=True, disabled=not can_proceed):
            if can_proceed:
                st.session_state.step = 3
                st.rerun()
        if not can_proceed:
            st.caption("⚠️ Please select or generate a dataset to continue.")

# ============================================================
# PAGE 3: AI MODEL ANALYSIS
# ============================================================
if st.session_state.step == 3:
    if not st.session_state.auth_verified:
        st.error("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 first.")
        if st.button("⬅️ Go Back to Step 1: Authorization", key="back_step3"):
            st.session_state.step = 1
            st.rerun()
        st.stop()
    if "df" not in st.session_state:
        st.error("⛔ ACCESS LOCKED: Please load or generate data on Step 2 first.")
        if st.button("⬅️ Go Back to Step 2: Data Input", key="back_step3_data"):
            st.session_state.step = 2
            st.rerun()
        st.stop()

    st.header("🤖 AI Model Training & Anomaly Detection")

    df = st.session_state.df
    contamination_rate = st.slider("Expected Anomaly Sensitivity Rate (%):", 1, 20, 7) / 100.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ignore_words = ["id", "employeeid", "staffid", "recordnumber", "bank_account"]
    feature_cols = [c for c in numeric_cols if c.lower() not in ignore_words]

    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0)
    X = X[X.columns[X.nunique() > 1]]

    if len(X) < 10:
        st.error("At least 10 records required for analysis.")
    else:
        X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = IsolationForest(n_estimators=200, contamination=contamination_rate, random_state=42)
        model.fit(X_train_scaled)

        all_preds = model.predict(scaler.transform(X))
        results = df.copy()
        results["anomalystatus"] = np.where(all_preds == -1, "Flagged for Review", "No Anomaly Flag")
        results["flagged"] = all_preds == -1
        results["anomalyscore"] = model.decision_function(scaler.transform(X))

        pca = PCA(n_components=2, random_state=42)
        scaled_full = scaler.transform(X)
        pcs = pca.fit_transform(scaled_full)
        results["pc1"] = pcs[:, 0]
        results["pc2"] = pcs[:, 1]

        st.session_state.results = results
        st.session_state.flagged_records = results[results["flagged"]].copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Analyzed", len(results))
        m2.metric("Flagged", len(st.session_state.flagged_records))
        m3.metric("Normal", len(results) - len(st.session_state.flagged_records))
        m4.metric("Percentage", f"{round(len(st.session_state.flagged_records) / len(results) * 100, 2)}%")

        st.subheader("🚩 Flagged Payroll Anomalies")
        st.dataframe(st.session_state.flagged_records, use_container_width=True)

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to Data Input", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_b2:
        if st.button("Continue to Visualizations & Reports ➡️", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

# ============================================================
# PAGE 4: VISUALIZATIONS & REPORTS
# ============================================================
if st.session_state.step == 4:
    if not st.session_state.auth_verified:
        st.error("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 first.")
        if st.button("⬅️ Go Back to Step 1: Authorization", key="back_step4_auth"):
            st.session_state.step = 1
            st.rerun()
        st.stop()
    if "results" not in st.session_state:
        st.error("⛔ ACCESS LOCKED: Please run AI Model Analysis on Step 3 first.")
        if st.button("⬅️ Go Back to Step 3: AI Analysis", key="back_step4_ai"):
            st.session_state.step = 3
            st.rerun()
        st.stop()

    st.header("📊 Advanced Payroll Analytics & Forensic Graphs")

    results = st.session_state.results
    flagged_records = st.session_state.flagged_records

    st.subheader("1. Dimensional Reduction & Regional Risk Analysis")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.pyplot(chart_pca(results))
    with col_v2:
        st.pyplot(chart_lga(results))

    st.divider()

    st.subheader("2. Anomaly Counts & Proportion Share")
    col_v3, col_v4 = st.columns(2)
    with col_v3:
        st.pyplot(chart_status_bar(results))
    with col_v4:
        st.pyplot(chart_status_pie(results))

    st.divider()

    st.subheader("3. Distributional Depth & Financial Correlation Analysis")
    col_v5, col_v6 = st.columns(2)
    with col_v5:
        st.pyplot(chart_salary_dist(results))
    with col_v6:
        st.pyplot(chart_allowance_scatter(results))

    st.divider()
    st.subheader("📦 Download Audit Package")
    summary_df = pd.DataFrame({
        "ReportItem": ["System", "DateTime", "Investigator", "Contact", "TotalRecords", "FlaggedRecords"],
        "Value": ["Federal Republic of Nigeria Payroll System", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), st.session_state.full_name, st.session_state.contact, len(results), len(flagged_records)]
    })

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nigeria_payroll_summary.csv", summary_df.to_csv(index=False))
        zf.writestr("full_results.csv", results.to_csv(index=False))
        zf.writestr("flagged_records.csv", flagged_records.to_csv(index=False))

    st.download_button(
        label="📥 Download Official Audit ZIP Package",
        data=zip_buffer.getvalue(),
        file_name=f"Nigeria_Payroll_Audit_{datetime.now().strftime('%Y%m%d')}.zip",
        mime="application/zip",
        use_container_width=True
    )

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to AI Analysis", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col_b2:
        if st.button("Continue to Q&A & Kudos ➡️", use_container_width=True):
            st.session_state.step = 5
            st.rerun()

# ============================================================
# PAGE 5: Q&A ASSISTANT & KUDOS
# ============================================================
if st.session_state.step == 5:
    if not st.session_state.auth_verified or "results" not in st.session_state:
        st.error("⛔ ACCESS LOCKED: Please complete previous mandatory steps first.")
        if st.button("⬅️ Go Back to Step 1", key="back_step5_lock"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    st.header("❓ Project Q & A Assistant & User Kudos")

    q_input = st.text_input("Ask a question about the system:")
    if q_input:
        ans = "I do not have a prepared answer for that question."
        if "algorithm" in q_input.lower() or "isolation" in q_input.lower():
            ans = "The system uses Isolation Forest, an unsupervised machine learning algorithm."
        elif "training" in q_input.lower():
            ans = "Training uses 80% of the dataset to learn standard payroll patterns."
        elif "report" in q_input.lower():
            ans = "The report contains summary statistics, full data results, and flagged anomaly tables packaged into a ZIP file."
        st.info("Assistant Response: " + ans)

    st.divider()
    st.subheader("🌟 Leave Your Kudos")
    with st.form("kudos_form"):
        k_name = st.text_input("Your Name or Department:", value=st.session_state.full_name)
        k_rating = st.select_slider("Rate System Performance:", options=["Poor", "Fair", "Good", "Very Good", "Excellent"], value="Excellent")
        k_comment = st.text_area("Your Kudos or Comments:", value="Great application!")
        submitted = st.form_submit_button("Submit Kudos")
        if submitted:
            st.success("🎉 Thank you! Your kudos have been logged successfully.")

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to Visualizations", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col_b2:
        if st.button("Continue to Developers ➡️", use_container_width=True):
            st.session_state.step = 6
            st.rerun()

# ============================================================
# PAGE 6: DEVELOPERS TEAM PAGE SLIDESHOW (12 MEMBERS)
# ============================================================
if st.session_state.step == 6:
    if not st.session_state.auth_verified or "results" not in st.session_state:
        st.error("⛔ ACCESS LOCKED: Please complete previous mandatory steps first.")
        if st.button("⬅️ Go Back to Step 1", key="back_step6_lock"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    st.header("👥 Meet the Development Team (12 Members)")
    st.markdown("Browse through the team members who built this forensic audit platform.")

    developers = [
        {
            "name": "A. O. JUWON",
            "address": "Ejuku, Yagba East LGA",
            "state": "Kogi State",
            "education": "NCE, B.Sc., M.Sc. (In View)",
            "role": "Team Lead and Project Problem Analyzer",
            "marital_status": "Married",
            "image": "developer 1.jpeg"
        },
        {
            "name": "ADEYEMI MARY MARVELOUS",
            "address": "ODO-ERE, YAGBA WEST LG",
            "state": "KOGI STATE",
            "education": "B. Sc",
            "role": "AI TEAM",
            "marital_status": "MARRIED",
            "image": "developer 2.jpeg"
        },
        {
            "name": "IBITOYE MAYOMIKUN JONATHAN",
            "address": "ISANLU, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B.Sc (ENG) ILORIN",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "Single",
            "image": "developer 3.jpeg"
        },
        {
            "name": "OLUKAIYE OLUGBENGA TIMOTHY",
            "address": "BOLORUNDURO KABBA, KABBA BUNU LG",
            "state": "KOGI STATE",
            "education": "B.Sc. (ENGINEERING), CUSTECH Osara",
            "role": "PROBLEM SOLVING TEAM",
            "marital_status": "Single",
            "image": "developer 4.jpeg"
        },
        {
            "name": "KOMOLAFE BABATUNDE RICHARD",
            "address": "AYETORO GBEDE IJUMU LG",
            "state": "KOGI STATE",
            "education": "B. Sc.",
            "role": "DEMONSTRATION TEAM",
            "marital_status": "Single",
            "image": "developer 5.jpeg"
        },
        {
            "name": "ADEWUMI BISIRIYU TUNDE",
            "address": "IFE_OLUKOTUN, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B. Sc, CNA, NBA (in view)",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "MARRIED",
            "image": "developer 6.jpeg"
        },
        {
            "name": "BABALOLA MARVELOUS ITUNUOLUWA",
            "address": "EGBE, YAGBA WEST",
            "state": "KOGI STATE",
            "education": "B. Sc (NURSING) FED.UNI LOKOJA",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "Single",
            "image": "developer 7.jpeg"
        },
        {
            "name": "AKANKU TAIYE OLUWAPEMISIRE",
            "address": "ITEDO ISANLU, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B Sc. (Econs)",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "MARRIED",
            "image": "developer 8.jpeg"
        },
        {
            "name": "VICTOR IWALAYE",
            "address": "IYARA, IJUMU LG",
            "state": "KOGI STATE",
            "education": "NATIONAL DEPLOMA",
            "role": "AI TEAM",
            "marital_status": "Single",
            "image": "developer 9.jpeg"
        },
        {
            "name": "RACHEAL OLORUNMAIYE (Shellie)",
            "address": "KABBA BUNNU, LG",
            "state": "KOGI STATE",
            "education": "B.Sc",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "N/A",
            "image": "developer 10.jpeg"
        },
        {
            "name": "ISREAL ADEOLA OLUMAIYEGUN",
            "address": "ODO ERE, YAGBA WEST LG",
            "state": "KOGI STATE",
            "education": "B. Sc. (Econs) KSU, kabba",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "Single",
            "image": "developer 11.jpeg"
        },
        {
            "name": "OLUWATOSIN MOSES O",
            "address": "ODO-ERE, YAGBA WEST",
            "state": "KOGI STATE",
            "education": "B.Sc., (Sociology)",
            "role": "AI TEAM",
            "marital_status": "MARRIED",
            "image": "developer 12.jpeg"
        }
    ]

    col_prev, col_mid, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("⬅️ Previous Developer", use_container_width=True):
            st.session_state.dev_index = (st.session_state.dev_index - 1) % len(developers)
            st.rerun()

    with col_next:
        if st.button("Next Developer ➡️", use_container_width=True):
            st.session_state.dev_index = (st.session_state.dev_index + 1) % len(developers)
            st.rerun()

    current_dev = developers[st.session_state.dev_index]

    st.markdown("---")
    c1, c2 = st.columns([1, 2])

    with c1:
        img_file = current_dev["image"]
        img_alt = img_file.replace(".jpeg", ".jpg") if ".jpeg" in img_file else img_file.replace(".jpg", ".jpeg")

        if os.path.exists(img_file):
            st.image(img_file, use_column_width=True)
        elif os.path.exists(img_alt):
            st.image(img_alt, use_column_width=True)
        else:
            st.info(f"🖼️ [Image file '{current_dev['image']}' not found in folder. Please add your JPEG picture.]")

    with c2:
        st.subheader(current_dev["name"])
        st.markdown(f"📍 **Address:** {current_dev['address']}")
        st.markdown(f"🏛️ **State:** {current_dev['state']}")
        st.markdown(f"🎓 **Education:** {current_dev['education']}")
        st.markdown(f"💼 **Role:** {current_dev['role']}")
        st.markdown(f"💍 **Marital Status:** {current_dev['marital_status']}")
        st.caption(f"Developer {st.session_state.dev_index + 1} of {len(developers)}")

    st.markdown("")
    if st.button("⬅️ Back to Q&A Assistant & Kudos", use_container_width=True):
        st.session_state.step = 5
        st.rerun()

# Footer
st.markdown("---")
st.caption("Federal Republic of Nigeria • Official Portal")
