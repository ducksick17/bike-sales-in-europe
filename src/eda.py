"""Reusable Plotly figures for exploring the Bike Sales in Europe dataset."""

import pandas as pd
import plotly.express as px

PALETTE = px.colors.qualitative.Set2
AGE_GROUP_ORDER = ["Youth (<25)", "Young Adults (25-34)", "Adults (35-64)", "Seniors (64+)"]
SAMPLE_SIZE = 3000  # scatter plots use a sample so the browser stays fast


def fig_profit_distribution(df: pd.DataFrame):
    return px.histogram(
        df, x="Profit", nbins=60, marginal="box", log_y=True,
        title="Distribution of Profit per Order (log scale)",
        color_discrete_sequence=PALETTE,
    )


def fig_category_split(df: pd.DataFrame):
    revenue = df.groupby("Product_Category")["Revenue"].sum().reset_index()
    return px.pie(
        revenue, names="Product_Category", values="Revenue", title="Revenue Share by Product Category",
        color_discrete_sequence=PALETTE,
    )


def fig_profit_by_category(df: pd.DataFrame):
    return px.box(
        df, x="Product_Category", y="Profit", color="Product_Category", log_y=True,
        title="Profit by Product Category (log scale)",
        color_discrete_sequence=PALETTE,
    )


def fig_profit_by_country(df: pd.DataFrame):
    grouped = df.groupby("Country")["Profit"].sum().sort_values(ascending=False).reset_index()
    return px.bar(
        grouped, x="Country", y="Profit", color="Country",
        title="Total Profit by Country",
        color_discrete_sequence=PALETTE,
    )


def fig_profit_vs_price(df: pd.DataFrame):
    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)
    return px.scatter_3d(
        sample, x="Unit_Price", y="Order_Quantity", z="Profit", color="Product_Category",
        title="Profit vs Unit Price vs Quantity (by Category)",
        color_discrete_sequence=PALETTE,
    )


def fig_profit_vs_age(df: pd.DataFrame):
    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)
    return px.scatter(
        sample, x="Customer_Age", y="Profit", color="Customer_Gender", opacity=0.6,
        title="Profit vs Customer Age (by Gender)",
        color_discrete_sequence=PALETTE,
    )


def fig_monthly_trend(df: pd.DataFrame):
    grouped = df.groupby(["Year", "Month_Number"])[["Revenue", "Profit"]].sum().reset_index()
    grouped["Period"] = pd.to_datetime(
        grouped["Year"].astype(str) + "-" + grouped["Month_Number"].astype(str) + "-01"
    )
    return px.line(
        grouped, x="Period", y=["Revenue", "Profit"],
        title="Monthly Revenue and Profit Over Time",
        color_discrete_sequence=PALETTE,
    )


def fig_profit_by_age_group(df: pd.DataFrame):
    grouped = df.groupby("Age_Group")["Profit"].mean().reset_index()
    return px.bar(
        grouped, x="Age_Group", y="Profit",
        title="Average Profit per Order by Age Group",
        category_orders={"Age_Group": AGE_GROUP_ORDER},
        color_discrete_sequence=PALETTE,
    )


def fig_margin_by_sub_category(df: pd.DataFrame):
    grouped = df.groupby(["Product_Category", "Sub_Category"])["Profit_Margin"].mean().reset_index()
    grouped = grouped.sort_values("Profit_Margin")
    return px.bar(
        grouped, x="Profit_Margin", y="Sub_Category", color="Product_Category", orientation="h",
        title="Average Profit Margin (%) by Sub-Category", height=600,
        color_discrete_sequence=PALETTE,
    )


def fig_correlation_heatmap(df: pd.DataFrame):
    numeric = df[["Customer_Age", "Order_Quantity", "Unit_Cost", "Unit_Price", "Month_Number", "Profit"]].copy()
    numeric["Is_Male"] = (df["Customer_Gender"] == "M").astype(int)
    corr = numeric.corr()
    return px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
    )
