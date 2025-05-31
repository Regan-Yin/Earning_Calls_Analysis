# dash_app.py

import os
import json
import pandas as pd

from dash import Dash, html, dcc, Input, Output
import plotly.express as px

# ------------------------------------------------------------------------------
# 1. File paths (point to your report/ folder)
# ------------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # FinBERT_Project/
REPORT_DIR  = os.path.join(BASE_DIR, "report")            # FinBERT_Project/report

METADATA_PATH           = os.path.join(BASE_DIR, "metadata.json")
EVENT_STUDY_CSV_PATH    = os.path.join(REPORT_DIR, "event_study.csv")
SENTIMENT_CSV_PATH      = os.path.join(REPORT_DIR, "sentiment_summary.csv")
PRICE_CHANGE_CSV_PATH   = os.path.join(REPORT_DIR, "price_change.csv")
CLASSIFICATION_TXT_PATH = os.path.join(REPORT_DIR, "classification_report.txt")
GRANGER_TXT_PATH        = os.path.join(REPORT_DIR, "granger.txt")

# ------------------------------------------------------------------------------
# 2. Load metadata.json and build a sorted list of tickers
# ------------------------------------------------------------------------------

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

all_tickers = sorted(
    {
        info["ticker"]
        for info in metadata.values()
        if isinstance(info, dict) and "ticker" in info
    }
)

# ------------------------------------------------------------------------------
# 3. Load CSVs / TXT files (from report/)
# ------------------------------------------------------------------------------

# Event‐study DataFrame (CAR)
event_df = pd.read_csv(EVENT_STUDY_CSV_PATH)

# Sentiment summary DataFrame
sentiment_df = pd.read_csv(SENTIMENT_CSV_PATH)

# Price‐change DataFrame
price_df = pd.read_csv(PRICE_CHANGE_CSV_PATH)

# Classification report (plain text)
with open(CLASSIFICATION_TXT_PATH, "r") as f:
    classification_text = f.read().strip()

# Granger causality results (plain text)
with open(GRANGER_TXT_PATH, "r") as f:
    granger_text = f.read().strip()

# ------------------------------------------------------------------------------
# 4. Preprocessing: Convert event_date to datetime, create "month" column, sort
# ------------------------------------------------------------------------------

for df in [event_df, sentiment_df, price_df]:
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["month"] = df["event_date"].dt.strftime("%Y-%m")
    df.sort_values(by="event_date", inplace=True)

# ------------------------------------------------------------------------------
# 5. Instantiate Dash and define layout
# ------------------------------------------------------------------------------

app = Dash(__name__)
app.title = "Earnings Call Dashboard"

app.layout = html.Div(
    [
        # ===== Header =====
        html.H1(
            "Earnings Call Dashboard",
            style={
                "textAlign": "center",
                "marginBottom": "20px",
                "marginTop": "20px",
                "fontFamily": "Arial, sans-serif",
            },
        ),

        # ===== Company Dropdown =====
        html.Div(
            [
                html.Label(
                    "Select Company (or multiple):",
                    style={"fontWeight": "bold", "marginRight": "10px"},
                ),
                dcc.Dropdown(
                    id="company-dropdown",
                    options=[{"label": t, "value": t} for t in all_tickers]
                    + [{"label": "All", "value": "All"}],
                    value=[all_tickers[0]],
                    multi=True,
                    placeholder="Type or select a ticker...",
                    style={"width": "300px"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "marginBottom": "30px",
            },
        ),

        # ===== Tabs =====
        dcc.Tabs(
            id="tabs",
            value="tab-overview",
            children=[
                dcc.Tab(label="Overview",              value="tab-overview"),
                dcc.Tab(label="CAR Analysis",          value="tab-car"),
                dcc.Tab(label="Sentiment Summary",     value="tab-sentiment"),
                dcc.Tab(label="Price Change",          value="tab-price"),
                dcc.Tab(label="Sentiment vs Price",    value="tab-scatter"),
                dcc.Tab(label="Classification Report", value="tab-classification"),
                dcc.Tab(label="Granger Causality",     value="tab-granger"),
            ],
            style={"marginBottom": "20px", "fontFamily": "Arial, sans-serif"},
        ),

        # ===== Content goes here =====
        html.Div(id="tabs-content", style={"padding": "0 20px"}),
    ],
    style={"fontFamily": "Arial, sans-serif"},
)


# ------------------------------------------------------------------------------
# 6. Callback: render content based on selected tab & tickers
# ------------------------------------------------------------------------------

@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
    Input("company-dropdown", "value"),
)
def render_tab_content(selected_tab, selected_tickers):
    """
    1) selected_tab: which tab is active
    2) selected_tickers: list of tickers chosen (or ["All"])
    Returns: a layout (div) specific to each tab, filtering DataFrames accordingly.
    """

    # If user picks "All", show data for all tickers
    if isinstance(selected_tickers, list) and "All" in selected_tickers:
        tickers_to_plot = all_tickers
    elif isinstance(selected_tickers, str) and selected_tickers == "All":
        tickers_to_plot = all_tickers
    else:
        if not selected_tickers:
            return html.Div(
                "⚠️ Please select at least one company from the dropdown above.",
                style={"fontStyle": "italic", "marginTop": "20px", "textAlign": "center"},
            )
        tickers_to_plot = (
            selected_tickers
            if isinstance(selected_tickers, list)
            else [selected_tickers]
        )

    # Filter each DataFrame by ticker(s)
    filtered_event     = event_df[event_df["ticker"].isin(tickers_to_plot)].copy()
    filtered_sentiment = sentiment_df[sentiment_df["ticker"].isin(tickers_to_plot)].copy()
    filtered_price     = price_df[price_df["ticker"].isin(tickers_to_plot)].copy()

    # Ensure they remain sorted by event_date
    filtered_event     = filtered_event.sort_values(by="event_date")
    filtered_sentiment = filtered_sentiment.sort_values(by="event_date")
    filtered_price     = filtered_price.sort_values(by="event_date")

    # ===== Tab: Overview =====
    if selected_tab == "tab-overview":
        overview_df = (
            filtered_event[["transcript", "ticker", "event_date"]]
            .sort_values(by=["ticker", "event_date"])
            .reset_index(drop=True)
        )
        overview_df["event_date"] = overview_df["event_date"].dt.strftime("%Y-%m-%d")

        return html.Div(
            [
                html.H3("Event Overview", style={"marginBottom": "15px", "textAlign": "center"}),
                html.P(
                    "Below is a table of all earnings‐call events (transcript keys + dates) "
                    "for the selected company(ies).",
                    style={"marginBottom": "10px", "textAlign": "center"},
                ),
                html.Table(
                    # Header row
                    [html.Tr([html.Th(col) for col in overview_df.columns])] +
                    # Data rows
                    [
                        html.Tr(
                            [html.Td(overview_df.iloc[i][col]) for col in overview_df.columns]
                        )
                        for i in range(len(overview_df))
                    ],
                    style={
                        "width": "60%",
                        "margin": "20px auto",
                        "borderCollapse": "collapse",
                        "border": "1px solid #ccc",
                        "textAlign": "left",
                    },
                ),
            ],
            style={"marginTop": "20px"},
        )

    # ===== Tab: CAR Analysis (Event‐Study) =====
    elif selected_tab == "tab-car":
        if filtered_event.empty:
            return html.Div(
                "No CAR data available for the selected ticker(s).",
                style={"textAlign": "center", "marginTop": "20px"},
            )

        # Plot CAR vs. month, grouped by ticker
        fig_car = px.line(
            filtered_event,
            x="month",
            y="CAR",
            color="ticker",
            markers=True,
            title="Cumulative Abnormal Return (CAR) Over Months",
            labels={"month": "Month (YYYY-MM)", "CAR": "CAR (%)"},
        )
        fig_car.update_layout(
            xaxis=dict(categoryorder="category ascending"),
            xaxis_tickangle=-45,
            margin={"t": 50, "b": 100},
            height=500,
            legend_title_text="Ticker",
        )

        return html.Div(
            [
                html.H3("CAR Analysis", style={"marginBottom": "15px", "textAlign": "center"}),
                dcc.Graph(figure=fig_car, style={"width": "90%", "margin": "auto"}),
            ],
            style={"marginTop": "20px"},
        )

    # ===== Tab: Sentiment Summary =====
    elif selected_tab == "tab-sentiment":
        if filtered_sentiment.empty:
            return html.Div(
                "No sentiment data available for the selected ticker(s).",
                style={"textAlign": "center", "marginTop": "20px"},
            )

        # Bar chart: sentiment_index vs. month, grouped by ticker
        fig_sent = px.bar(
            filtered_sentiment,
            x="month",
            y="sentiment_index",
            color="ticker",
            barmode="group",
            title="Sentiment Index of Earnings Calls by Month",
            labels={"month": "Month (YYYY-MM)", "sentiment_index": "Sentiment Index"},
        )
        fig_sent.update_layout(
            xaxis=dict(categoryorder="category ascending"),
            xaxis_tickangle=-45,
            margin={"t": 50, "b": 100},
            height=500,
            legend_title_text="Ticker",
        )

        return html.Div(
            [
                html.H3("Sentiment Summary", style={"marginBottom": "15px", "textAlign": "center"}),
                dcc.Graph(figure=fig_sent, style={"width": "90%", "margin": "auto"}),
            ],
            style={"marginTop": "20px"},
        )

    # ===== Tab: Price Change =====
    elif selected_tab == "tab-price":
        if filtered_price.empty:
            return html.Div(
                "No price‐change data available for the selected ticker(s).",
                style={"textAlign": "center", "marginTop": "20px"},
            )

        # Bar chart: price_change_pct vs. month, grouped by ticker
        fig_price = px.bar(
            filtered_price,
            x="month",
            y="price_change_pct",
            color="ticker",
            barmode="group",
            title="Price Change Percentage (Post vs. Pre Earnings) by Month",
            labels={"month": "Month (YYYY-MM)", "price_change_pct": "Price Change %"},
        )
        fig_price.update_layout(
            xaxis=dict(categoryorder="category ascending"),
            xaxis_tickangle=-45,
            margin={"t": 50, "b": 100},
            height=500,
            legend_title_text="Ticker",
        )

        # Build a small HTML table of price details
        price_table_df = filtered_price[
            ["ticker", "event_date", "close_before", "close_after", "price_change_pct"]
        ].copy()
        price_table_df["event_date"] = price_table_df["event_date"].dt.strftime("%Y-%m-%d")
        price_table_df = price_table_df.sort_values(by=["ticker", "event_date"]).reset_index(drop=True)

        return html.Div(
            [
                html.H3("Price Change Analysis", style={"marginBottom": "15px", "textAlign": "center"}),
                dcc.Graph(figure=fig_price, style={"width": "90%", "margin": "auto"}),
                html.Br(),
                html.Div(
                    [
                        html.P(
                            "Detailed Price‐Before and Price‐After Table:",
                            style={"fontWeight": "bold", "marginBottom": "10px", "textAlign": "center"},
                        ),
                        html.Table(
                            [html.Tr([html.Th(col) for col in price_table_df.columns])] +
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            round(price_table_df.iloc[i][col], 4)
                                            if isinstance(price_table_df.iloc[i][col], float)
                                            else price_table_df.iloc[i][col]
                                        )
                                        for col in price_table_df.columns
                                    ]
                                )
                                for i in range(len(price_table_df))
                            ],
                            style={
                                "width": "80%",
                                "margin": "20px auto",
                                "borderCollapse": "collapse",
                                "border": "1px solid #ccc",
                                "textAlign": "left",
                            },
                        ),
                    ],
                    style={"marginTop": "20px"},
                ),
            ],
            style={"marginTop": "20px"},
        )

    # ===== Tab: Sentiment vs Price (Scatter + Regression) =====
    elif selected_tab == "tab-scatter":
        # Merge sentiment + price on ticker + event_date
        merged = pd.merge(
            filtered_sentiment[["ticker", "event_date", "sentiment_index"]],
            filtered_price[["ticker", "event_date", "price_change_pct"]],
            on=["ticker", "event_date"],
            how="inner"
        )

        if merged.empty:
            return html.Div(
                "No overlapping sentiment & price‐change data available for the selected ticker(s).",
                style={"textAlign": "center", "marginTop": "20px"},
            )

        # Scatter with OLS trendline (one overall line)
        fig_scatter = px.scatter(
            merged,
            x="sentiment_index",
            y="price_change_pct",
            color="ticker",
            trendline="ols",
            trendline_scope="overall",
            title="Scatter: Sentiment Index vs. Price Change (%)",
            labels={"sentiment_index": "Sentiment Index", "price_change_pct": "Price Change %"},
        )
        fig_scatter.update_layout(
            margin={"t": 50, "b": 50},
            height=500,
            legend_title_text="Ticker",
        )

        return html.Div(
            [
                html.H3(
                    "Sentiment vs. Price Change (with OLS Regression Line)",
                    style={"marginBottom": "15px", "textAlign": "center"},
                ),
                dcc.Graph(figure=fig_scatter, style={"width": "80%", "margin": "auto"}),
            ],
            style={"marginTop": "20px"},
        )

    # ===== Tab: Classification Report =====
    elif selected_tab == "tab-classification":
        return html.Div(
            [
                html.H3("Classification Report (Logistic Regression)", style={"textAlign": "center"}),
                html.Pre(
                    classification_text,
                    style={
                        "whiteSpace": "pre-wrap",
                        "wordBreak": "break-all",
                        "backgroundColor": "#f5f5f5",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "margin": "auto",
                        "width": "80%",
                        "border": "1px solid #ddd",
                        "fontSize": "14px",
                    },
                ),
            ],
            style={"marginTop": "20px"},
        )

    # ===== Tab: Granger Causality =====
    elif selected_tab == "tab-granger":
        lines = [l.strip() for l in granger_text.splitlines() if l.strip()]
        table_rows = []
        for line in lines:
            if "lag=" in line and "p-value=" in line:
                parts = line.replace("lag=", "").replace("p-value=", "").split()
                lag = parts[0]
                pval = parts[1]
                table_rows.append({"lag": lag, "p-value": pval})

        return html.Div(
            [
                html.H3("Granger Causality Test Results", style={"textAlign": "center"}),
                html.Br(),
                html.Table(
                    [html.Tr([html.Th("Lag"), html.Th("P-Value")])] +
                    [html.Tr([html.Td(r["lag"]), html.Td(r["p-value"])]) for r in table_rows],
                    style={
                        "width": "40%",
                        "margin": "auto",
                        "borderCollapse": "collapse",
                        "border": "1px solid #ccc",
                        "textAlign": "left",
                    },
                ),
            ],
            style={"marginTop": "20px"},
        )

    # ===== Fallback =====
    else:
        return html.Div("This tab is currently under construction.")


# ------------------------------------------------------------------------------
# 7. Run the server
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # On Dash ≥ v3.x, use `app.run` (instead of `app.run_server`)
    app.run(debug=True)
