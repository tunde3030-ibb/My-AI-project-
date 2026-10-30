"""
AGENCY BANKING AND FINANCIAL INCLUSION
Yagba East Local Government Area, Kogi State, Nigeria

A Streamlit application for analysing questionnaire data.
- Upload a real CSV or Excel questionnaire dataset.
- Explore respondent demographics and agency-banking usage.
- Summarise Likert-scale responses.
- Examine relationships between agency-banking factors and financial inclusion.
- Download a blank data-entry template.

IMPORTANT:
This application does not create or claim real research findings. Use data collected
from your respondents. The downloadable template contains blank/example structure
only; replace it with your actual field data before drawing conclusions.

Run:
    pip install streamlit pandas numpy matplotlib openpyxl
    streamlit run agency_banking_financial_inclusion.py

Suggested column names:
    Location, Gender, Age_Group, Used_Agency_Banking,
    Accessibility, Availability, Convenience, Affordability,
    Reliability, Financial_Inclusion

Likert responses should be coded from 1 to 5:
    1 = Strongly Disagree, 2 = Disagree, 3 = Neutral,
    4 = Agree, 5 = Strongly Agree
"""

import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Agency Banking & Financial Inclusion",
    page_icon="🏦",
    layout="wide"
)

LOCATIONS = ["Isanlu", "Ejuku", "Ponyan", "Ife-Olukotun", "Alu-Igbagun"]
LIKERT_COLUMNS = [
    "Accessibility",
    "Availability",
    "Convenience",
    "Affordability",
    "Reliability",
    "Financial_Inclusion",
]
LIKERT_LABELS = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly Agree",
}


def make_template():
    """Return a blank CSV template with the suggested questionnaire fields."""
    columns = [
        "Respondent_ID",
        "Location",
        "Gender",
        "Age_Group",
        "Education_Level",
        "Occupation",
        "Used_Agency_Banking",
        "Usage_Frequency",
        "Accessibility",
        "Availability",
        "Convenience",
        "Affordability",
        "Reliability",
        "Financial_Inclusion",
    ]
    return pd.DataFrame(columns=columns)


def normalise_yes_no(series):
    """Standardise common yes/no entries for consistent counting."""
    mapping = {
        "yes": "Yes", "y": "Yes", "1": "Yes", "true": "Yes",
        "no": "No", "n": "No", "0": "No", "false": "No",
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(
        series.astype(str).str.strip()
    )


def to_numeric_likert(data, columns):
    """Convert Likert columns to numeric values, coercing invalid entries."""
    for col in columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data.loc[~data[col].between(1, 5), col] = np.nan
    return data


def display_metric(label, value):
    st.metric(label, value)


st.title("🏦 Agency Banking and Financial Inclusion")
st.subheader("Yagba East Local Government Area, Kogi State, Nigeria")
st.write(
    "A research-data dashboard for organising and analysing questionnaire "
    "responses on agency banking and financial inclusion."
)

with st.expander("Research variables and coding guide"):
    st.markdown(
        """
        **Agency-banking explanatory factors:** accessibility, availability of
        agents, convenience, affordability and reliability.

        **Financial-inclusion outcome:** the respondent's financial-inclusion
        score, based on the questionnaire items selected for that construct.

        **Likert coding:** 1 = Strongly Disagree, 2 = Disagree, 3 = Neutral,
        4 = Agree, 5 = Strongly Agree.

        Use the same scoring direction for all items. If a question is negatively
        worded, reverse-code it before calculating a combined score.
        """
    )

st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload questionnaire data",
    type=["csv", "xlsx", "xls"],
    help="Upload your collected responses in CSV or Excel format."
)

template = make_template()
st.sidebar.download_button(
    "Download blank data template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="agency_banking_data_template.csv",
    mime="text/csv",
)

if uploaded_file is None:
    st.info(
        "Upload your actual questionnaire dataset to begin. You can download "
        "the blank template from the sidebar."
    )
    st.markdown("### Suggested sample allocation")
    allocation = pd.DataFrame({
        "Location": LOCATIONS,
        "Planned Respondents": [40, 40, 40, 40, 40],
    })
    st.dataframe(allocation, use_container_width=True, hide_index=True)
    st.metric("Planned sample size", "200")
    st.warning(
        "No respondent results are displayed because no actual dataset has "
        "been uploaded. Do not use invented responses as research findings."
    )
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the uploaded file: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded file contains no records.")
    st.stop()

df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
df = to_numeric_likert(df, LIKERT_COLUMNS)

if "Used_Agency_Banking" in df.columns:
    df["Used_Agency_Banking"] = normalise_yes_no(df["Used_Agency_Banking"])

st.success(f"Dataset loaded: {len(df):,} respondent record(s), {len(df.columns)} column(s).")

# Overview
st.header("1. Dataset overview")
c1, c2, c3 = st.columns(3)
with c1:
    display_metric("Respondents / records", f"{len(df):,}")
with c2:
    display_metric("Variables", f"{len(df.columns):,}")
with c3:
    missing = int(df.isna().sum().sum())
    display_metric("Missing data cells", f"{missing:,}")

with st.expander("View uploaded data"):
    st.dataframe(df, use_container_width=True)

# Data validation
st.header("2. Data quality checks")
quality = pd.DataFrame({
    "Column": df.columns,
    "Missing values": [int(df[c].isna().sum()) for c in df.columns],
    "Missing (%)": [
        round(float(df[c].isna().mean() * 100), 1) for c in df.columns
    ],
    "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
})
st.dataframe(quality, use_container_width=True, hide_index=True)

if "Location" in df.columns:
    st.header("3. Respondent distribution by location")
    location_counts = (
        df["Location"].astype(str).str.strip().value_counts()
        .reindex(LOCATIONS, fill_value=0)
    )
    other_locations = df["Location"].astype(str).str.strip().value_counts()
    other_locations = other_locations[~other_locations.index.isin(LOCATIONS)]
    if not other_locations.empty:
        location_counts = pd.concat([location_counts, other_locations])
    st.bar_chart(location_counts)
    location_table = location_counts.rename_axis("Location").reset_index(name="Respondents")
    location_table["Percentage"] = (
        location_table["Respondents"] / max(len(df), 1) * 100
    ).round(1)
    st.dataframe(location_table, use_container_width=True, hide_index=True)

# Demographics
st.header("4. Respondent profile")
demographic_cols = [
    c for c in ["Gender", "Age_Group", "Education_Level", "Occupation"]
    if c in df.columns
]
if demographic_cols:
    tabs = st.tabs(demographic_cols)
    for tab, col in zip(tabs, demographic_cols):
        with tab:
            counts = df[col].fillna("Not stated").astype(str).value_counts()
            st.bar_chart(counts)
            st.dataframe(
                counts.rename_axis(col).reset_index(name="Frequency"),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.caption("No standard demographic columns were found in the uploaded data.")

# Usage
st.header("5. Agency-banking usage")
if "Used_Agency_Banking" in df.columns:
    usage = df["Used_Agency_Banking"].fillna("Not stated").value_counts()
    left, right = st.columns(2)
    with left:
        st.bar_chart(usage)
    with right:
        usage_table = usage.rename_axis("Used agency banking").reset_index(name="Frequency")
        usage_table["Percentage"] = (
            usage_table["Frequency"] / max(len(df), 1) * 100
        ).round(1)
        st.dataframe(usage_table, use_container_width=True, hide_index=True)
    st.caption(
        "Only respondents who have used agency banking should answer questions "
        "that ask about their personal experience. Keep non-user experience items blank."
    )
else:
    st.caption("Add a 'Used_Agency_Banking' column (Yes/No) to analyse usage.")

if "Usage_Frequency" in df.columns:
    st.subheader("Frequency of use")
    st.bar_chart(df["Usage_Frequency"].fillna("Not stated").astype(str).value_counts())

# Likert descriptive statistics
st.header("6. Descriptive analysis of research variables")
available_likert = [c for c in LIKERT_COLUMNS if c in df.columns]
if available_likert:
    summary = df[available_likert].agg(["count", "mean", "std", "min", "max"]).T
    summary = summary.rename(columns={
        "count": "Valid responses",
        "mean": "Mean",
        "std": "Standard deviation",
        "min": "Minimum",
        "max": "Maximum",
    })
    summary["Mean"] = summary["Mean"].round(2)
    summary["Standard deviation"] = summary["Standard deviation"].round(2)
    st.dataframe(summary, use_container_width=True)

    means = df[available_likert].mean().dropna()
    if not means.empty:
        st.subheader("Mean scores by variable")
        st.bar_chart(means)
        st.caption(
            "A higher mean indicates stronger agreement with the statements, "
            "provided all items are coded in the same direction."
        )
else:
    st.info(
        "No recognised Likert columns were found. Use the suggested column "
        "names or edit the column names in your dataset."
    )

# Composite score and relationship analysis
st.header("7. Relationship between agency banking and financial inclusion")
factor_cols = [
    c for c in [
        "Accessibility", "Availability", "Convenience",
        "Affordability", "Reliability"
    ] if c in df.columns
]

if "Financial_Inclusion" in df.columns and factor_cols:
    relationship = df[factor_cols + ["Financial_Inclusion"]].corr(
        method="pearson", min_periods=2
    )
    st.subheader("Pearson correlation matrix")
    st.dataframe(relationship.round(3), use_container_width=True)

    correlations = relationship["Financial_Inclusion"].drop("Financial_Inclusion")
    correlations = correlations.dropna()
    if not correlations.empty:
        st.subheader("Correlation with financial inclusion")
        st.bar_chart(correlations)
        st.caption(
            "Correlation ranges from -1 to +1. It describes association, not "
            "proof that one variable causes another."
        )
else:
    st.info(
        "For correlation analysis, include at least one agency-banking factor "
        "and a numeric 'Financial_Inclusion' score coded from 1 to 5."
    )

# Optional composite score
if factor_cols:
    st.header("8. Overall agency-banking score")
    df["Agency_Banking_Composite"] = df[factor_cols].mean(axis=1)
    st.metric(
        "Average agency-banking composite score",
        f"{df['Agency_Banking_Composite'].mean():.2f}"
        if df["Agency_Banking_Composite"].notna().any() else "Not available"
    )
    st.caption(
        "This is the row-wise average of the available agency-banking factor "
        "scores. Confirm that the questionnaire items measure compatible "
        "constructs before using a composite score in the thesis."
    )

# Export
st.header("9. Download analysis data")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download dataset with calculated composite score",
    data=csv_bytes,
    file_name="agency_banking_analysis_data.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Prepared as a research-support tool. Interpret results in line with your "
    "research objectives, hypotheses, sampling design and supervisor's guidance. "
    "The app does not automatically establish statistical significance or causation."
)
