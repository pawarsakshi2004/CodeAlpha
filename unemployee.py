import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="Unemployment Analysis India",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# CUSTOM LIGHT THEME
# ==================================================

st.markdown("""
<style>

.stApp{
    background-color:#F5F7FA;
}

[data-testid="stSidebar"]{
    background-color:#EAF2FF;
}

h1,h2,h3{
    color:#1E293B;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Unemployment.csv")

    df.columns = df.columns.str.strip()

    df.rename(columns={
        "Estimated Unemployment Rate (%)": "Unemployment_Rate",
        "Estimated Employed": "Employed",
        "Estimated Labour Participation Rate (%)":
            "Labour_Participation_Rate"
    }, inplace=True)

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df.dropna(subset=["Date"], inplace=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

    df["Covid_Period"] = np.where(
        df["Date"] >= pd.Timestamp("2020-03-01"),
        "During Covid",
        "Pre Covid"
    )

    return df


df = load_data()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📊 Dashboard Menu")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "State Analysis",
        "Covid Impact",
        "State Comparison",
        "Correlation Analysis",
        "Heatmap",
        "Policy Insights"
    ]
)

states = sorted(df["Region"].dropna().unique())

selected_state = st.sidebar.selectbox(
    "Select State",
    ["All States"] + list(states)
)

filtered_df = df.copy()

if selected_state != "All States":
    filtered_df = filtered_df[
        filtered_df["Region"] == selected_state
    ]

# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.title("📊 Unemployment Analysis in India")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Unemployment Rate",
        f"{df['Unemployment_Rate'].mean():.2f}%"
    )

    col2.metric(
        "Highest Unemployment Rate",
        f"{df['Unemployment_Rate'].max():.2f}%"
    )

    col3.metric(
        "Lowest Unemployment Rate",
        f"{df['Unemployment_Rate'].min():.2f}%"
    )

    st.subheader("Overall Unemployment Trend")

    trend = (
        df.groupby("Date")["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="Date",
        y="Unemployment_Rate",
        markers=True,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# STATE ANALYSIS
# ==================================================

elif page == "State Analysis":

    st.title("📈 State Analysis")

    trend = (
        filtered_df.groupby("Date")["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="Date",
        y="Unemployment_Rate",
        markers=True,
        title="Unemployment Trend",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonal Trend")

    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    seasonal = (
        filtered_df.groupby("Month")["Unemployment_Rate"]
        .mean()
        .reindex(month_order)
        .fillna(0)
        .reset_index()
    )

    fig2 = px.line(
        seasonal,
        x="Month",
        y="Unemployment_Rate",
        markers=True,
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# COVID IMPACT
# ==================================================

elif page == "Covid Impact":

    st.title("🦠 Covid-19 Impact Analysis")

    covid = (
        filtered_df.groupby("Covid_Period")
        ["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        covid,
        x="Covid_Period",
        y="Unemployment_Rate",
        color="Covid_Period",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    if len(covid) == 2:

        pre = covid.loc[
            covid["Covid_Period"] == "Pre Covid",
            "Unemployment_Rate"
        ].values[0]

        during = covid.loc[
            covid["Covid_Period"] == "During Covid",
            "Unemployment_Rate"
        ].values[0]

        if pre != 0:
            change = ((during - pre) / pre) * 100

            st.metric(
                "Covid Impact (%)",
                f"{change:.2f}%"
            )

    st.subheader("State-wise Covid Impact")

    covid_state = (
        df.groupby(["Region", "Covid_Period"])
        ["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        covid_state,
        x="Region",
        y="Unemployment_Rate",
        color="Covid_Period",
        barmode="group"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# STATE COMPARISON
# ==================================================

elif page == "State Comparison":

    st.title("🔥 State Comparison")

    top_states = (
        df.groupby("Region")["Unemployment_Rate"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_states,
        x="Region",
        y="Unemployment_Rate",
        color="Unemployment_Rate",
        template="plotly_white",
        title="Top 10 States by Unemployment Rate"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Employment vs Unemployment")

    fig2 = px.scatter(
        filtered_df,
        x="Employed",
        y="Unemployment_Rate",
        size="Labour_Participation_Rate",
        hover_name="Region",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# CORRELATION ANALYSIS
# ==================================================

elif page == "Correlation Analysis":

    st.title("🔍 Correlation Analysis")

    corr = filtered_df[
        [
            "Unemployment_Rate",
            "Employed",
            "Labour_Participation_Rate"
        ]
    ].corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ==================================================
# HEATMAP
# ==================================================

elif page == "Heatmap":

    st.title("🌡️ State-wise Heatmap")

    pivot = df.pivot_table(
        values="Unemployment_Rate",
        index="Region",
        columns="Year",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot,
        annot=True,
        cmap="Reds",
        fmt=".1f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

    st.info(
        "Darker regions indicate higher unemployment levels."
    )

# ==================================================
# POLICY INSIGHTS
# ==================================================

elif page == "Policy Insights":

    st.title("📑 Policy Insights & Recommendations")

    state_avg = (
        df.groupby("Region")["Unemployment_Rate"]
        .mean()
    )

    highest_state = state_avg.idxmax()
    highest_rate = state_avg.max()

    st.subheader("Key Findings")

    st.write(f"""
    • **{highest_state}** recorded the highest average
      unemployment rate of **{highest_rate:.2f}%**.

    • Covid-19 caused a significant increase in unemployment.

    • Seasonal fluctuations indicate unemployment varies
      throughout the year.

    • Labour participation and unemployment are related.

    • Some states consistently show higher unemployment
      than others and may require targeted intervention.
    """)

    st.subheader("Policy Recommendations")

    st.markdown("""
    1. Increase employment generation programs in states
       with persistently high unemployment.

    2. Strengthen social protection measures during
       economic shocks and pandemics.

    3. Expand skill development and vocational training.

    4. Promote MSMEs and entrepreneurship to create jobs.

    5. Improve labour participation through education and
       workforce inclusion initiatives.

    6. Design state-specific employment policies based on
       unemployment patterns.
    """)