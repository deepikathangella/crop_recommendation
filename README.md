# Crop Recommendation with Model Confidence and Out-of-Range Detection

## Files
| File | Purpose |
|------|---------|
| `Crop_recommendation_5000.csv` | Dataset (5000 rows, 64 crops) |
| `train_model.py` | Steps 3-10: preprocessing, EDA, model comparison, evaluation, tuning, saves the model |
| `crop_model.pkl` | Saved model + out-of-range detector (created by train_model.py) |
| `predictor.py` | Loads the model, checks input, gives recommendation + confidence + unusual flag |
| `app.py` | FastAPI backend (`/predict`, `/health`, serves the web page) |
| `static/index.html`, `style.css` | Web page |
| `static/validation.js` | Input checking ("Invalid input" for words, symbols, empty, out of range) |
| `static/script.js` | Sends the form to the backend and shows the result |
| `outputs/` | EDA plots and result tables (CSV) for your report |

## How to run
```
pip install -r requirements.txt
python train_model.py        # takes about 2-3 minutes, creates crop_model.pkl
uvicorn app:app --reload     # starts the server
```
Open http://127.0.0.1:8000 in the browser (this is the local URL for step 16).
API documentation page: http://127.0.0.1:8000/docs

## How the recommendation is decided
1. Every input must be a number inside the allowed range, otherwise "Invalid input".
2. Unusual check: if a value is outside the training min/max, or the combination is far from
   its 5 nearest training rows, no crop is recommended.
3. Confidence check: if the top probability is below 50%, the result is shown as "not sure".
4. Otherwise the crop is recommended with its confidence and the top 3 crops.
