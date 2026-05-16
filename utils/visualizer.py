"""
visualizer.py
Các hàm vẽ biểu đồ Plotly cho TLU Feedback Analytics.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

COLORS = {
    "Tích cực 😄": "#10b981",
    "Tiêu cực 😠": "#ef4444",
    "Trung lập 😐": "#eab308",
    "Positive": "#10b981",
    "Negative": "#ef4444",
    "Neutral": "#eab308",
}

def sentiment_donut(df_sentiments: pd.Series, title: str = "Phân bố Cảm xúc") -> go.Figure:
    """Biểu đồ donut tổng quan sentiment."""
    counts = df_sentiments.value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        counts, values="Count", names="Sentiment", hole=0.6,
        color="Sentiment",
        color_discrete_map=COLORS,
        title=title,
    )
    fig.update_layout(
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def aspect_bar_chart(df: pd.DataFrame, aspect_cols: list[str]) -> go.Figure:
    """Biểu đồ cột grouped theo aspect và sentiment."""
    bar_data = []
    for col in aspect_cols:
        aspect_name = col.replace("_Sentiment", "").replace("_", " ")
        counts = df[col].value_counts().to_dict()
        for sent, count in counts.items():
            if pd.notna(sent):
                bar_data.append({"Khía cạnh": aspect_name, "Cảm xúc": sent, "Số lượng": count})

    if not bar_data:
        return go.Figure()

    df_bar = pd.DataFrame(bar_data)
    fig = px.bar(
        df_bar, x="Khía cạnh", y="Số lượng", color="Cảm xúc",
        barmode="group",
        color_discrete_map=COLORS,
        title="Phân tích Cảm xúc theo Khía cạnh",
    )
    fig.update_layout(
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
    )
    return fig


def confidence_bar(distribution: list, title: str = "") -> go.Figure:
    """Biểu đồ ngang hiển thị phân phối xác suất của 3 class."""
    labels = ["Tiêu cực", "Trung lập", "Tích cực"]
    colors_list = ["#ef4444", "#eab308", "#10b981"]
    fig = go.Figure(go.Bar(
        x=distribution,
        y=labels,
        orientation='h',
        marker_color=colors_list,
        text=[f"{v:.1%}" for v in distribution],
        textposition='outside',
    ))
    fig.update_layout(
        title=title,
        margin=dict(t=30, b=5, l=5, r=60),
        height=160,
        xaxis=dict(range=[0, 1.1], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig


def trend_line_chart(df: pd.DataFrame, date_col: str, sentiment_col: str) -> go.Figure:
    """Biểu đồ xu hướng sentiment theo thời gian (nếu có cột ngày)."""
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, sentiment_col])
    if df.empty:
        return go.Figure()

    df['Month'] = df[date_col].dt.to_period('M').astype(str)
    grouped = df.groupby(['Month', sentiment_col]).size().reset_index(name='Count')
    fig = px.line(
        grouped, x='Month', y='Count', color=sentiment_col,
        color_discrete_map=COLORS,
        markers=True,
        title="Xu hướng Cảm xúc theo Thời gian",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig
