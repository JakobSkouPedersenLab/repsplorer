"""3D PCA explorer of the learned representation. Run: python app.py -> http://127.0.0.1:8050"""
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dcc, html, no_update
from sklearn.decomposition import PCA

# ---- Data + PCA (once at startup) ----
reps = pd.read_csv("train_reps.tsv", sep="\t", index_col="sample_id")
anno = pd.read_csv("train_anno.tsv", sep="\t", index_col="sample_id").loc[reps.index]

pca = PCA(n_components=3)
df = pd.DataFrame(pca.fit_transform(reps), columns=["PC1", "PC2", "PC3"], index=reps.index)
df = df.join(anno[["cancer_type", "primary_site", "color"]]).reset_index()
var = pca.explained_variance_ratio_ * 100

# Cancer types have their own colour; a primary site takes the colour of its most common cancer type.
ct_color = dict(zip(df.cancer_type, df.color))
site_color = df.groupby("primary_site").cancer_type.agg(lambda s: s.mode()[0]).map(ct_color).to_dict()
MODES = {
    "cancer_type": ("Cancer type", ct_color),
    "primary_site": ("Primary site", site_color),
}

# Fixed axis ranges so the axes don't jump when groups are toggled.
pad = 0.05 * (df[["PC1", "PC2", "PC3"]].max() - df[["PC1", "PC2", "PC3"]].min())
RANGES = {pc: [df[pc].min() - pad[pc], df[pc].max() + pad[pc]] for pc in ["PC1", "PC2", "PC3"]}


def groups(mode):
    """Group names for a label mode, largest first."""
    return df[mode].value_counts().index.tolist()


def options(mode):
    counts, colors = df[mode].value_counts(), MODES[mode][1]
    return [
        {
            "label": html.Span([
                html.Span("●", style={"color": colors[g], "fontSize": "1.3em", "margin": "0 6px"}),
                f"{g} ({counts[g]})",
            ]),
            "value": g,
        }
        for g in counts.index
    ]


# ---- Layout ----
app = Dash(__name__, title="Representation Explorer")
app.layout = html.Div(
    style={"display": "flex", "fontFamily": "Helvetica, Arial, sans-serif", "height": "100vh", "margin": 0},
    children=[
        html.Div(style={"flex": 3, "padding": "10px"}, children=[
            html.H2("Explore the learned representation in 3D", style={"margin": "4px 0"}),
            html.P(
                "Each dot is a tumour sample. Samples from the same tissue cluster together, "
                "even though the model never saw the labels. Drag to rotate, scroll to zoom.",
                style={"margin": "4px 0", "color": "#555"},
            ),
            dcc.Graph(
                id="plot",
                style={"height": "88vh"},
                config={"displaylogo": False, "scrollZoom": True},
            ),
        ]),
        html.Div(
            style={"flex": 1, "padding": "10px", "borderLeft": "1px solid #ddd",
                   "display": "flex", "flexDirection": "column", "minWidth": "240px"},
            children=[
                html.H4("Label by", style={"margin": "8px 0"}),
                dcc.RadioItems(
                    id="mode",
                    options=[{"label": f" {name}", "value": m} for m, (name, _) in MODES.items()],
                    value="cancer_type",
                    labelStyle={"display": "block", "fontSize": "1.1em", "margin": "4px 0"},
                ),
                html.Div(style={"margin": "12px 0", "display": "flex", "gap": "8px"}, children=[
                    html.Button("Select all", id="all", style={"flex": 1, "padding": "8px"}),
                    html.Button("Clear", id="clear", style={"flex": 1, "padding": "8px"}),
                ]),
                # One static checklist per mode, shown/hidden: Dash 4 crashes when a callback
                # replaces options whose labels are components (the coloured swatches).
                *[
                    dcc.Checklist(
                        id=f"groups-{m}",
                        options=options(m),
                        value=groups(m),
                        labelStyle={"display": "block", "margin": "3px 0"},
                        style={"overflowY": "auto", "flex": 1, "display": "block" if m == "cancer_type" else "none"},
                    )
                    for m in MODES
                ],
            ],
        ),
    ],
)


# ---- Callbacks ----
@app.callback(
    *[Output(f"groups-{m}", "style") for m in MODES],
    *[Output(f"groups-{m}", "value") for m in MODES],
    Input("mode", "value"),
    Input("all", "n_clicks"),
    Input("clear", "n_clicks"),
    prevent_initial_call=True,
)
def update_checklist(mode, _all, _clear):
    styles = [{"overflowY": "auto", "flex": 1, "display": "block" if m == mode else "none"} for m in MODES]
    if ctx.triggered_id == "mode":  # switching label type: show everything of the new type
        values = [groups(m) for m in MODES]
    else:
        values = [([] if ctx.triggered_id == "clear" else groups(m)) if m == mode else no_update for m in MODES]
    return *styles, *values


@app.callback(
    Output("plot", "figure"),
    Input("mode", "value"),
    *[Input(f"groups-{m}", "value") for m in MODES],
)
def update_figure(mode, *values):
    name, colors = MODES[mode]
    selected = set(dict(zip(MODES, values))[mode] or [])
    fig = go.Figure()
    for g in groups(mode):
        if g not in selected:
            continue
        d = df[df[mode] == g]
        fig.add_trace(go.Scatter3d(
            x=d.PC1, y=d.PC2, z=d.PC3,
            mode="markers",
            name=g,
            marker={"size": 3, "color": colors[g], "opacity": 0.8},
            customdata=d[["sample_id", "cancer_type", "primary_site"]],
            hovertemplate="%{customdata[0]}<br>Cancer type: %{customdata[1]}"
                          "<br>Primary site: %{customdata[2]}<extra></extra>",
        ))
    axis = lambda i: {"title": f"PC{i + 1} ({var[i]:.1f}%)", "range": RANGES[f"PC{i + 1}"]}
    fig.update_layout(
        template="plotly_white",
        uirevision="keep",  # keep the camera where the visitor left it
        margin={"l": 0, "r": 0, "t": 40, "b": 0},  # top margin keeps the toolbar off the legend
        legend={"title": {"text": name}, "itemsizing": "constant"},
        scene={"xaxis": axis(0), "yaxis": axis(1), "zaxis": axis(2), "aspectmode": "cube"},
    )
    return fig


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
