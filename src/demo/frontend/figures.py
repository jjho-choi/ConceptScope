import numpy as np
import plotly.express as px


def plot_concept_bar(df, num_concepts=20, height=600):

    subset = df[:num_concepts]
    num_target = len(subset[subset["Concept Type"] == "target"])
    color_map = {"target": "green", "context": "orange", "bias": "red"}
    bias_threshold = subset["bias_threshold"].iloc[0]
    fig = px.bar(
        subset,
        x="slice_idx",
        y="Mean",
        color="Concept Type",
        color_discrete_map=color_map,
        hover_data={
            "Mean": True,
            "latent_name": True,
            "latent_idx": True,
            "Concept Type": True,
            "Class aligned": False,  # hide duplicate
        },
    )

    fig.update_layout(
        xaxis_title="Concepts",
        yaxis_title="Concept Strength",
        xaxis_tickangle=-30,
        xaxis=dict(
            tickmode="array",
            tickvals=subset["slice_idx"],
            ticktext=subset["latent_name"],
        ),
        yaxis_title_font=dict(size=18, family="Arial", color="black"),  # increase size
        yaxis=dict(tickfont=dict(size=14), nticks=5),  # Limit to 5 y-axis ticks
        template="plotly_white",
        font=dict(size=14),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.add_shape(
        type="line",
        x0=num_target - 0.5,
        x1=len(subset) - 1,
        y0=bias_threshold,
        y1=bias_threshold,
        line=dict(color="red", width=2, dash="dash"),
    )
    return fig
