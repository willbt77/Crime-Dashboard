import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load full crime dataset
@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_police_data.csv")

# Load Cambridgeshire summary data with population and crime rate
@st.cache_data
def load_aggregated_data():
    return pd.read_csv("Cambridgeshire_Crime_Dataset_with_Extrapolated_Income__2023_.csv")

@st.cache_data
def load_model_data():
    return pd.read_csv("Model_Ready_Crime_Dataset.csv")  


# Load data
crime_data = load_data()
aggr_data = load_aggregated_data()

# Title
st.title("Disposable Incomes relationship with crime in Cambridgeshire (2023)")
st.markdown("""  
This dashboard explores the relationship between disposable income levels and crime across local authorities in Cambridgeshire, using real population and crime data.  
Below includes trends, crime distributions, and predictions based on regional income and population.

The data shows that there is an inverse relation between income and crime, the following visualisations will provide reference to this
""")


# Sidebar Filters
la_options = aggr_data['Mapped LA'].unique()
selected_la = st.sidebar.selectbox("Select Local Authority", la_options)

# Filtered Data
filtered_data = crime_data[crime_data['LSOA name'].str.startswith(selected_la)]

# Summary Metrics in 3 columns
with st.container():
    st.subheader("Key Statistics for Selected Region")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Crimes (2023)", value=f"{aggr_data[aggr_data['Mapped LA'] == selected_la]['Total Crimes'].values[0]:,}")
    with col2:
        st.metric(label="Avg. Disposable Income (2023)", value=f"£{aggr_data[aggr_data['Mapped LA'] == selected_la]['Income_2023'].values[0]:,}")
    with col3:
        st.metric(label="Crime Rate / 1,000", value=f"{aggr_data[aggr_data['Mapped LA'] == selected_la]['Crime_Rate_per_1000'].values[0]:.2f}")

# Crime Type Distribution
crime_type_counts = filtered_data['Crime type'].value_counts().reset_index()
crime_type_counts.columns = ['Crime Type', 'Count']
fig1 = px.bar(crime_type_counts, x='Crime Type', y='Count', title=f"Crime Type Distribution in {selected_la}")
st.plotly_chart(fig1)

# Map of Crime Locations
fig2 = px.scatter_mapbox(filtered_data,
                         lat="Latitude",
                         lon="Longitude",
                         color="Crime type",
                         zoom=9,
                         height=500,
                         title=f"Crime Locations in {selected_la} (2023)")
fig2.update_layout(mapbox_style="open-street-map")
fig2.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig2)

# Monthly Crime Trend
crime_data['Date'] = pd.to_datetime(crime_data['Date'], dayfirst=True, errors='coerce')
filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], dayfirst=True, errors='coerce')
filtered_data_valid = filtered_data[filtered_data['Date'].notna()]

if not filtered_data_valid.empty:
    monthly_trend = (
        filtered_data_valid
        .groupby(filtered_data_valid['Date'].dt.to_period("M"))
        .size()
        .reset_index(name='Crimes')
    )
    monthly_trend['Date'] = monthly_trend['Date'].dt.strftime('%Y-%m')
    fig3 = px.line(monthly_trend, x='Date', y='Crimes', title=f"Monthly Crime Trend in {selected_la} (2023)")
    st.plotly_chart(fig3)
else:
    st.info("No valid date data available to display monthly crime trends.")

# Scatter Plot: Income vs Crime Rate per LA
st.subheader("Income vs Crime Rate per 1,000 Residents")
st.markdown("""
This plot visualizes how disposable income levels may relate to crime rates.  
Each point represents a local authority. The trendline helps identify whether lower income areas generally experience higher crime per capita.
""")
fig4 = px.scatter(
    aggr_data,
    x='Income_2023',
    y='Crime_Rate_per_1000',
    text='Mapped LA',
    trendline='ols',
    labels={
        'Income_2023': 'Average Income (2023)',
        'Crime_Rate_per_1000': 'Crime Rate per 1,000 Residents'
    },
    title="Do Lower Disposable Incomes Relate to Higher Crime Rates?"
)
fig4.update_traces(textposition='top center')
st.plotly_chart(fig4)

# Load UK-wide dataset for modeling
model_data = load_model_data()

# Predictive Model: High Crime Risk Classifier
st.subheader("Predict Crime Risk Based on Disposable Income and Population (UK-wide)")
st.markdown("""
Using a machine learning model trained on UK-wide data, it classified each region as high or low crime risk based on two main factors:  
- Average disposable income (2023, extrapolated from past data)  
- Population size (2023)  

Note: Crime is influenced by many complex social and economic factors. This model only highlights general trends, therefore cannot attribute income to be the main driver in crime.
""")

# Train model
features = ['Income_2023', 'Population_2023']
X = model_data[features]
y = model_data['High_Crime']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")

# Input for prediction
st.markdown("### Predict Crime Risk for a Custom Region")
input_income = st.number_input("Enter Average Disposable Income (£)", min_value=10000, max_value=50000, value=25000)
input_population = st.number_input("Enter Population", min_value=10000, max_value=300000, value=100000)

input_df = pd.DataFrame({
    'Income_2023': [input_income],
    'Population_2023': [input_population]
})

prediction = model.predict(input_df)[0]
pred_label = "High Crime Risk" if prediction == 1 else "Low Crime Risk"
st.success(f"Predicted Category: {pred_label}")
