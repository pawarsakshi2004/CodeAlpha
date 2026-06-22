import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Iris.csv")

# =========================
# DATA PREPARATION
# =========================

X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# MODEL EVALUATION
# =========================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =========================
# SIDEBAR NAVIGATION
# =========================

st.sidebar.title("🌸 Iris Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Analysis",
        "Visualizations",
        "Model Performance",
        "Predict Species",
        "About Project"
    ]
)

# =========================
# HOME PAGE
# =========================

if page == "Home":

    st.title("🌸 Iris Flower Classification")

    st.write("""
    Welcome to the Iris Flower Classification Project.

    This machine learning application predicts the species of an Iris flower based on:

    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width

    Species:
    - Iris Setosa
    - Iris Versicolor
    - Iris Virginica
    """)

    st.success("Machine Learning Model: Random Forest Classifier")

# =========================
# DATASET ANALYSIS PAGE
# =========================

elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum().reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}
    ))

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

# =========================
# VISUALIZATIONS PAGE
# =========================

elif page == "Visualizations":

    st.title("📈 Data Visualizations")

    st.subheader("Species Distribution")

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    sns.countplot(
        x="Species",
        data=df,
        palette="Set2",
        ax=ax1
    )

    plt.xticks(rotation=15)

    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")

    numeric_df = df.drop(["Id", "Species"], axis=1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax2
    )

    st.pyplot(fig2)

# =========================
# MODEL PERFORMANCE PAGE
# =========================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    st.success(
        f"Model Accuracy: {accuracy * 100:.2f}%"
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance,
        palette="viridis",
        ax=ax3
    )

    st.pyplot(fig3)

# =========================
# PREDICTION PAGE
# =========================

elif page == "Predict Species":

    st.title("🔍 Predict Iris Species")

    st.write("Move the sliders and click Predict.")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        4.0, 8.0, 5.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        2.0, 5.0, 3.5
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        1.0, 7.0, 1.4
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        0.1, 3.0, 0.2
    )

    if st.button("Predict Species"):

        prediction = model.predict([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        species = encoder.inverse_transform(prediction)

        st.success(
            f"Predicted Species: {species[0]}"
        )

# =========================
# ABOUT PAGE
# =========================

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.write("""
    ### Project Name
    Iris Flower Classification


    ### Objective
    To classify Iris flowers into different species using Machine Learning.

    ### Algorithm Used
    Random Forest Classifier

    ### Libraries Used
    - Pandas
    - Seaborn
    - Matplotlib
    - Scikit-Learn
    - Streamlit

    ### Dataset Features
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width

    ### Target Variable
    Species
    """)

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("Developed by Sakshi Pawar for CodeAlpha Data Science Internship")