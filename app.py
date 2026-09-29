"""3D PCA explorer of the learned representation. Run: python app.py -> http://127.0.0.1:8050"""
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from sklearn.decomposition import PCA

# ---- Data + PCA (once at startup) ----
reps = pd.read_csv("train_reps.tsv", sep="\t", index_col="sample_id")
anno = pd.read_csv("train_anno.tsv", sep="\t", index_col="sample_id").loc[reps.index]

N_PCS = 5
PCS = [f"PC{i + 1}" for i in range(N_PCS)]

pca = PCA(n_components=N_PCS)
df = pd.DataFrame(pca.fit_transform(reps), columns=PCS, index=reps.index)
df = df.join(anno[["cancer_type", "primary_site", "color"]]).reset_index()
var = pca.explained_variance_ratio_ * 100

# Cancer types have their own colour; a primary site takes the colour of its most common cancer type.
ct_color = dict(zip(df.cancer_type, df.color))
site_color = df.groupby("primary_site").cancer_type.agg(lambda s: s.mode()[0]).map(ct_color).to_dict()
# Each cancer_type belongs to exactly one primary_site (a site can have several types, e.g.
# Kidney = KIRC + KIRP + KICH). Used to carry a selection across the Label-by toggle.
type_to_site = df.groupby("cancer_type").primary_site.first().to_dict()
MODES = {
    "cancer_type": ("Cancer type", ct_color),
    "primary_site": ("Primary site", site_color),
}

# Fixed axis ranges so the axes don't jump when groups or PCs are changed.
pad = 0.05 * (df[PCS].max() - df[PCS].min())
RANGES = {pc: [df[pc].min() - pad[pc], df[pc].max() + pad[pc]] for pc in PCS}


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
            dcc.Store(id="camera"),
        ]),
        html.Div(
            style={"flex": 1, "padding": "10px", "borderLeft": "1px solid #ddd",
                   "display": "flex", "flexDirection": "column", "minWidth": "240px"},
            children=[
                html.H4("Axes", style={"margin": "8px 0"}),
                html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "12px"}, children=[
                    html.Div(style={"flex": 1}, children=[
                        html.Label(axis_label, style={"fontSize": "0.85em", "color": "#555"}),
                        dcc.Dropdown(
                            id=f"axis-{axis_label}",
                            options=[{"label": f"{pc} ({var[i]:.0f}%)", "value": pc} for i, pc in enumerate(PCS)],
                            value=default,
                            clearable=False,
                        ),
                    ])
                    for axis_label, default in zip(["X", "Y", "Z"], PCS[:3])
                ]),
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
# Plotly's own uirevision doesn't reliably survive Dash's Plotly.react() calls, so the camera
# is captured client-side (no server round trip) and re-applied explicitly on every rebuild.
app.clientside_callback(
    """
    function(relayoutData) {
        if (relayoutData && relayoutData["scene.camera"]) {
            return relayoutData["scene.camera"];
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("camera", "data"),
    Input("plot", "relayoutData"),
)


@app.callback(
    Output("axis-X", "options"),
    Output("axis-Y", "options"),
    Output("axis-Z", "options"),
    Input("axis-X", "value"),
    Input("axis-Y", "value"),
    Input("axis-Z", "value"),
)
def update_axis_options(x_pc, y_pc, z_pc):
    """Each dropdown offers every PC except the ones already taken by the other two."""
    chosen = {"axis-X": x_pc, "axis-Y": y_pc, "axis-Z": z_pc}
    def opts(axis_id):
        taken = {v for k, v in chosen.items() if k != axis_id}
        return [{"label": f"{pc} ({var[i]:.0f}%)", "value": pc} for i, pc in enumerate(PCS) if pc not in taken]
    return opts("axis-X"), opts("axis-Y"), opts("axis-Z")


@app.callback(
    *[Output(f"groups-{m}", "style") for m in MODES],
    *[Output(f"groups-{m}", "value") for m in MODES],
    Input("mode", "value"),
    Input("all", "n_clicks"),
    Input("clear", "n_clicks"),
    State("groups-cancer_type", "value"),
    State("groups-primary_site", "value"),
    prevent_initial_call=True,
)
def update_checklist(mode, _all, _clear, ct_value, site_value):
    styles = [{"overflowY": "auto", "flex": 1, "display": "block" if m == mode else "none"} for m in MODES]
    if ctx.triggered_id == "mode":
        # Carry the selection across the toggle via the real type<->site mapping, instead of
        # resetting to "everything": type->site collapses (union of sites of checked types);
        # site->type expands (a checked site pulls in all of its cancer types).
        if mode == "primary_site":
            new_site_value = sorted({type_to_site[ct] for ct in (ct_value or [])})
            values = [no_update, new_site_value]
        else:
            selected_sites = set(site_value or [])
            new_ct_value = [ct for ct in groups("cancer_type") if type_to_site[ct] in selected_sites]
            values = [new_ct_value, no_update]
    else:
        values = [([] if ctx.triggered_id == "clear" else groups(m)) if m == mode else no_update for m in MODES]
    return *styles, *values


@app.callback(
    Output("plot", "figure"),
    Input("mode", "value"),
    Input("axis-X", "value"),
    Input("axis-Y", "value"),
    Input("axis-Z", "value"),
    *[Input(f"groups-{m}", "value") for m in MODES],
    State("camera", "data"),
)
def update_figure(mode, x_pc, y_pc, z_pc, *values_and_camera):
    *values, camera = values_and_camera
    name = MODES[mode][0]
    selected = set(dict(zip(MODES, values))[mode] or [])
    fig = go.Figure()
    # Always emit one trace per group in BOTH modes (a fixed 58 traces), toggling `visible`
    # rather than adding/removing traces: Plotly fully rebuilds the WebGL scene -- and resets
    # the camera -- whenever the trace COUNT changes, which used to happen on every checkbox
    # click or mode switch.
    for m, (_, colors) in MODES.items():
        for g in groups(m):
            d = df[df[m] == g]
            fig.add_trace(go.Scatter3d(
                x=d[x_pc], y=d[y_pc], z=d[z_pc],
                mode="markers",
                name=g,
                visible=(m == mode) and (g in selected),
                marker={"size": 3, "color": colors[g], "opacity": 0.8},
                customdata=d[["sample_id", "cancer_type", "primary_site"]],
                hovertemplate="%{customdata[0]}<br>Cancer type: %{customdata[1]}"
                              "<br>Primary site: %{customdata[2]}<extra></extra>",
            ))
    axis = lambda pc: {"title": f"{pc} ({var[PCS.index(pc)]:.1f}%)", "range": RANGES[pc]}
    fig.update_layout(
        template="plotly_white",
        uirevision="keep",  # keep the camera where the visitor left it
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=True,  # Plotly hides the legend by default when only one group is visible
        legend={
            "title": {"text": name},
            "itemsizing": "constant",
            "x": 0.85,
            "xanchor": "left",
        },
        scene={"xaxis": axis(x_pc), "yaxis": axis(y_pc), "zaxis": axis(z_pc),
               "aspectmode": "cube", "dragmode": "orbit",
               # A 3D scene ignores layout.margin (unlike a 2D axis it never auto-shrinks to
               # make room for the legend), so its domain is narrowed here instead -- a fixed
               # fraction of the paper, reserving a real, constant strip on the right that the
               # legend can sit in without ever overlapping the data, in either label mode.
               "domain": {"x": [0, 0.83], "y": [0, 1]},
               **({"camera": camera} if camera else {})},
    )
    return fig


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
