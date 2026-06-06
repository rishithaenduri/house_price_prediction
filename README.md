# 🏙️ Bangalore Property Price Prediction — ML Project

## Project Structure
```
bangalore-price-prediction/
├── house_price_prediction.ipynb   # Jupyter Notebook (full ML pipeline)
├── app.py                         # Flask backend API
├── static/
│   └── index.html                 # Frontend UI
├── model/                         # Auto-generated after running notebook
│   ├── house_price_model.pkl
│   ├── encoders.pkl
│   └── metadata.json
├── requirements.txt
└── README.md
```

## Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Jupyter Notebook
Open and run all cells in `house_price_prediction.ipynb`.  
This trains the model and saves files to the `model/` folder.

### Step 3 — Start the Flask server
```bash
python app.py
```

### Step 4 — Open the web app
Visit: http://localhost:5001

## ML Pipeline Summary

| Step | Description |
|------|-------------|
| Data | 2,500-row synthetic Bangalore property dataset |
| EDA  | Distribution plots, correlation heatmap, neighborhood analysis |
| Models | Linear Regression, Decision Tree, Random Forest, Gradient Boosting |
| Best Model | Gradient Boosting (highest R² = 0.9259) |
| CV | 5-fold cross-validation R² ≈ 0.9358 ± 0.007 |
| API | Flask REST endpoint `/api/predict` |
| Frontend | Vanilla HTML/CSS/JS with teal UI & live predictions |

## Key Changes vs Original
- **City**: Bangalore market with 10 real neighborhoods (Koramangala, Indiranagar, Whitefield…)
- **New features**: `Has_Balcony` and `Lift` (amenity toggles in UI)
- **Best model**: Gradient Boosting (was Random Forest) — R² 0.9259 vs 0.8945
- **4 models compared**: Added Decision Tree to the comparison
- **Cross-validation**: 5-fold CV scores reported
- **Confidence band**: ±10% (was ±8%)
- **Dataset**: 2,500 rows (was 2,000)
- **Port**: 5001

## API Endpoints
- `GET /api/metadata` — returns dropdown options, model R² and CV score
- `POST /api/predict` — accepts property features, returns predicted price

### Sample request body
```json
{
  "area": 1200,
  "bedrooms": 2,
  "bathrooms": 2,
  "age": 3,
  "parking": 1,
  "floor": 7,
  "has_balcony": 1,
  "lift": 1,
  "neighborhood": "Whitefield",
  "property_type": "Apartment",
  "furnishing": "Semi-furnished",
  "condition": "Good"
}
```

## Neighborhoods & Market Tier

| Neighborhood | Multiplier | Tier |
|---|---|---|
| Koramangala | 1.85× | Premium |
| Indiranagar | 1.75× | Premium |
| Jayanagar | 1.55× | Upper Mid |
| HSR Layout | 1.45× | Upper Mid |
| Whitefield | 1.35× | Mid |
| Marathahalli | 1.25× | Mid |
| Hebbal | 1.20× | Mid |
| JP Nagar | 1.15× | Standard |
| Sarjapur Road | 1.10× | Standard |
| Electronic City | 1.00× | Affordable |

## Tech Stack
Python · scikit-learn · pandas · Flask · HTML/CSS/JS
