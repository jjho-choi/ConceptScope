import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
        legend=dict(
            x=0.99,
            y=0.99,  # position inside plot (top-right)
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.5)",  # semi-transparent background
            bordercolor="black",
            borderwidth=1,
        ),
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


def plot_top_class_for_concept(latent_avg_activations, selected_class, class_names, top_k=5):
    latent_avg_activations = np.array(latent_avg_activations)
    sorted_indices = np.argsort(latent_avg_activations)[::-1]
    high_class_indices = sorted_indices[:top_k]
    top_values = latent_avg_activations[high_class_indices]
    top_class_names = [class_names[idx] for idx in high_class_indices]
    hover_text = [f"{idx}: {class_names[idx]}" for idx in high_class_indices]

    selected_class_idx = class_names.index(selected_class)

    if selected_class_idx in high_class_indices:
        class_rank = np.where(high_class_indices == selected_class_idx)[0][0] + 1
        class_value = latent_avg_activations[selected_class_idx]
    else:
        class_rank = np.where(sorted_indices == selected_class_idx)[0][0] + 1
        class_value = latent_avg_activations[selected_class_idx]

    info = f"ℹ️ **{selected_class}** class has a concept strength ranked #{class_rank} among all classes, with an activation value of {class_value:.2f}."

    bar_colors = ["orange" if idx == selected_class_idx else "#888" for idx in high_class_indices]

    fig = go.Figure(
        go.Bar(
            x=top_class_names,
            y=top_values,
            text=[f"{v:.2f}" for v in top_values],
            textposition="auto",
            marker=dict(color=bar_colors),
            hovertext=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        # title=f"Top-{top_k} Class Activations for Selected Concept",
        xaxis_title="Class Name",
        yaxis_title="Concept Strength",
        margin=dict(l=40, r=20, t=50, b=80),
        # height=VisualizationConfig.figure_height,
    )
    return fig, info
