"""Simple sales forecasting: monthly totals + Linear Regression."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def prepare_monthly(df, date_col, sales_col):
    """Step 1: clean the data and add up sales for each month."""
    data = df[[date_col, sales_col]].dropna().copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data["month"] = data[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = data.groupby("month")[sales_col].sum().reset_index()
    monthly.columns = ["month", "sales"]
    return monthly


def make_features(months, first_month):
    """Step 2: turn each month into numbers the model can use.
    - time    : how many months since the start (captures the trend)
    - 1..12   : which calendar month it is (captures seasonality)"""
    X = pd.get_dummies(months.dt.month).reindex(columns=range(1, 13), fill_value=0)
    X = X.astype(int)
    X["time"] = (months.dt.year - first_month.year) * 12 + (months.dt.month - first_month.month)
    X.columns = X.columns.astype(str)
    return X


def test_model(monthly, test_months=6):
    """Step 3: train on older months, test on the last 6 to check accuracy."""
    first = monthly["month"].iloc[0]
    X = make_features(monthly["month"], first)
    y = monthly["sales"]

    X_train, X_test = X[:-test_months], X[-test_months:]
    y_train, y_test = y[:-test_months], y[-test_months:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    error = mean_absolute_error(y_test, predicted)
    error_percent = error / y_test.mean() * 100
    test_result = pd.DataFrame({
        "month": monthly["month"][-test_months:],
        "Actual": y_test,
        "Predicted": predicted,
    })
    return test_result, error_percent


def forecast(monthly, months_ahead):
    """Step 4: train on ALL the data and predict the next few months."""
    first = monthly["month"].iloc[0]
    X = make_features(monthly["month"], first)

    model = LinearRegression()
    model.fit(X, monthly["sales"])

    last = monthly["month"].iloc[-1]
    future_months = pd.Series(
        pd.date_range(last + pd.DateOffset(months=1), periods=months_ahead, freq="MS")
    )
    future_X = make_features(future_months, first)
    return pd.DataFrame({"month": future_months, "sales": model.predict(future_X)})


def get_insights(monthly, future):
    """Step 5: plain-English insights using basic pandas."""
    insights = []

    best = monthly.loc[monthly["sales"].idxmax()]
    worst = monthly.loc[monthly["sales"].idxmin()]
    insights.append(f"Best month: {best['month']:%B %Y} ({best['sales']:,.0f})")
    insights.append(f"Worst month: {worst['month']:%B %Y} ({worst['sales']:,.0f})")

    # which calendar month is usually strongest?
    by_month = monthly.groupby(monthly["month"].dt.month_name())["sales"].mean()
    insights.append(f"Strongest month of the year on average: {by_month.idxmax()}")
    insights.append(f"Weakest month of the year on average: {by_month.idxmin()}")

    # compare first half vs second half of the data to describe the trend
    half = len(monthly) // 2
    first_avg = monthly["sales"][:half].mean()
    second_avg = monthly["sales"][half:].mean()
    change = (second_avg / first_avg - 1) * 100
    word = "increased" if change >= 0 else "decreased"
    insights.append(f"Average monthly sales {word} by {abs(change):.1f}% "
                    f"between the first half and the second half of the data")

    # forecast vs recent sales
    recent_avg = monthly["sales"][-6:].mean()
    future_avg = future["sales"].mean()
    change = (future_avg / recent_avg - 1) * 100
    word = "higher" if change >= 0 else "lower"
    insights.append(f"Forecast average is {abs(change):.1f}% {word} than the last 6 months")

    return insights
