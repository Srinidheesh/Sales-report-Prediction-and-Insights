import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from forecasting import prepare_monthly, test_model, forecast, get_insights

sns.set_theme(style="whitegrid", rc={"figure.figsize": (10, 4)})

st.title("Sales Prediction & Insights")
st.write("Upload a sales file (CSV or Excel) to see a forecast, charts and insights.")

# ---- 1. Upload the file ----
file = st.file_uploader("Upload sales file", type=["csv", "xlsx"])
if file is None:
    st.info("Upload a file to begin. You can try sample_sales.csv.")
    st.stop()

if file.name.endswith(".csv"):
    df = pd.read_csv(file)
else:
    df = pd.read_excel(file)

st.subheader("Data preview")
st.dataframe(df.head())

# ---- 2. Choose columns ----
date_col = st.selectbox("Which column is the date?", df.columns)
sales_col = st.selectbox("Which column is the sales amount?", df.columns, index=1)
months_ahead = st.slider("How many months to forecast?", 3, 12, 6)

monthly = prepare_monthly(df, date_col, sales_col)
if len(monthly) < 18:
    st.error("Please upload at least 18 months of data.")
    st.stop()

# ---- 3. Run the model ----
test_result, error_percent = test_model(monthly)
future = forecast(monthly, months_ahead)

# ---- 4. Charts ----
st.subheader("Sales trend and forecast")
actual = monthly.assign(Type="Actual")
predicted = future.assign(Type="Forecast")
chart_data = pd.concat([actual, predicted])
plt.figure()  # start a fresh blank chart
ax = sns.lineplot(data=chart_data, x="month", y="sales", hue="Type")
st.pyplot(ax.figure)

st.subheader("Average sales by calendar month")
season = monthly.assign(calendar_month=monthly["month"].dt.month)
plt.figure()  # start a fresh blank chart
ax = sns.barplot(data=season, x="calendar_month", y="sales")
st.pyplot(ax.figure)

st.subheader("Total sales per year")
yearly = monthly.assign(year=monthly["month"].dt.year)
plt.figure()  # start a fresh blank chart
ax = sns.barplot(data=yearly, x="year", y="sales", estimator="sum", errorbar=None)
st.pyplot(ax.figure)

# ---- 5. How accurate is the model? ----
st.subheader("How accurate is the model?")
st.write(f"On the last 6 months (which the model had not seen), "
         f"predictions were off by about **{error_percent:.1f}%** on average.")
test_long = test_result.melt(id_vars="month", var_name="Type", value_name="sales")
plt.figure()  # start a fresh blank chart
ax = sns.lineplot(data=test_long, x="month", y="sales", hue="Type", marker="o")
st.pyplot(ax.figure)

# ---- 6. Insights ----
st.subheader("Key insights")
for line in get_insights(monthly, future):
    st.write("- " + line)

st.subheader("Forecast table")
st.dataframe(future.round(0))
