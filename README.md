# Representation Explorer

Interactive 3D PCA (PC1–PC3) of `train_reps.tsv`, coloured and filtered by `train_anno.tsv`.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Open http://127.0.0.1:8050 and go full screen (Ctrl+Cmd+F on macOS, F11 elsewhere).

## Controls

- **Drag** to rotate, **scroll/pinch** to zoom, **double-click** to reset the view.
- **Label by**: colour and filter by cancer type or primary site.
- **Checkboxes** / **Select all** / **Clear**: choose which groups are shown.
- Click a legend entry to hide it; double-click to show only that one.
- Hover a dot to see its sample ID, cancer type and primary site.
