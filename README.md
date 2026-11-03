# KittenLens — cat breed web app

One-page website: upload (or drag, or paste) a cat photo and a fine-tuned EfficientNet-B0
predicts one of 12 breeds, shows the top-3 probabilities and a Grad-CAM "where did the AI look" map.

```
mushuklens-app/
├── app.py                 Flask backend (model loading, /api/predict, Grad-CAM)
├── static/index.html      the whole frontend (no external libraries, works offline)
├── model/                 put mushuklens_effnet_b0.pt here
├── notebooks/             training notebook (Google Colab)
└── requirements.txt
```

## 1. Get the trained weights (required)
This zip does **not** contain trained weights. Create them yourself:
1. Open `notebooks/MushukLens_fine_tuning.ipynb` in Google Colab, set Runtime to GPU, Run all.
2. The training cell saves `/content/model/mushuklens_effnet_b0.pt`.
3. Download it (Colab sidebar → Files → model → ⋮ → Download) and place it in this project's `model/` folder.

## 2. Run
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

Options (environment variables): `MODEL_PATH`, `HOST`, `PORT`.
If the weights file is missing the page still opens and shows a clear banner instead of fake predictions.

## API
- `GET  /api/status` → `{model_loaded, classes, device, error}`
- `POST /api/predict` (multipart field `image`) → `{predictions:[{breed, probability}×3], heatmap:"data:image/png;base64,…"}`

## Limits (honest notes)
- The model only knows 12 breeds; it will name one of them even for a dog, a fox or a plush toy.
- Low-confidence results (<50%) get a warning in the UI, but confidence is not a guarantee of correctness.
- Uploaded images are processed in memory only, never written to disk. Max 8 MB.
- `python app.py` uses Flask's built-in server, fine for a demo. For deployment use e.g. `gunicorn -w 1 app:app`
  (keep one worker per model copy; each loads its own).
