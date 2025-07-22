![GitHub repo size](https://img.shields.io/github/repo-size/hendifires/tsunami-prediction-system)
![GitHub last commit](https://img.shields.io/github/last-commit/hendifires/tsunami-prediction-system)
![GitHub license](https://img.shields.io/github/license/hendifires/tsunami-prediction-system)

# 🌊 Tsunami Prediction System

A machine learning-based web application for predicting tsunami risks triggered by both tectonic earthquakes and volcanic activity. This system utilizes ensemble classifiers and well-preprocessed geophysical datasets to provide reliable and fast tsunami predictions.

---

## 🚀 Key Features

- **Tectonic Tsunami Prediction**  
  Predicts tsunami risk using seismic parameters:
  - Earthquake Magnitude (`Mag`)
  - Focal Depth (km)
  - Latitude
  - Longitude

- **Volcanic Tsunami Vulnerability Prediction**  
  Assesses tsunami potential due to volcanic activity based on:
  - Earthquake Magnitude (`Eq`)
  - Elevation (m)
  - Latitude
  - Longitude
  - Volcano Type (`Type`)
  - Volcanic Explosivity Index (`VEI`)

---

## 🗂️ Project Structure

```
tsunami-pred/
├── app.py                   # FastAPI application entry point
├── requirements.txt         # Python dependencies
├── dataset/                 # Raw datasets
│   ├── dataset_tektonik.csv
│   └── dataset_vulkanik.csv
├── models/                  # Trained ML models and preprocessing artifacts
│   ├── ensemble_tek.joblib
│   ├── ensemble_vul.joblib
│   ├── encoder_vul.joblib
│   └── scaler_tek.joblib
├── pred_type/               # Data schema definitions
│   └── types.py
├── src/                     # Prediction logic
│   └── predict.py
└── static/                  # Web frontend interface
    └── index.html
```

---

## ⚙️ Installation Guide

1. **Clone the repository**  
   ```bash
   git clone <repository-url>
   cd tsunami-pred
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python -m venv venv
   # For Unix/MacOS:
   source venv/bin/activate
   # For Windows:
   venv\Scripts\activate
   ```

3. **Install project dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the App

1. **Start the FastAPI server**  
   ```bash
   uvicorn app:app --reload
   ```

2. **Open in your browser**  
   Visit: [http://localhost:8000](http://localhost:8000)

3. **Choose a prediction type**  
   - **Tectonic**: Input seismic parameters
   - **Volcanic**: Input volcano parameters

4. **Get your result**  
   After submitting the form, the app will display whether a tsunami is likely to occur.

---

## 📡 API Endpoints

### 🔹 Tectonic Prediction
- **Endpoint**: `/v1/predict/tektonik`  
- **Method**: `POST`  
- **Sample Request**:
```json
{
  "datas": [
    {
      "Mag": 6.7,
      "Focal_Depth_km": 10.5,
      "Latitude": -5.2,
      "Longitude": 120.3
    }
  ]
}
```

### 🔸 Volcanic Prediction
- **Endpoint**: `/v1/predict/vulnerability`  
- **Method**: `POST`  
- **Sample Request**:
```json
{
  "datas": [
    {
      "Eq": 5.8,
      "Elevation_m": 1350,
      "Latitude": -7.6,
      "Longitude": 112.8,
      "Type": "Stratovolcano",
      "VEI": 4
    }
  ]
}
```

---

## 🧠 Technical Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **ML Library**: scikit-learn (ensemble models with joblib export)
- **Data Processing**: pandas, StandardScaler, OneHotEncoder
- **Frontend**: Simple static HTML interface
- **Deployment-ready**: Uvicorn ASGI server with CORS support

---

## 📊 Datasets

| File Name              | Description                         |
|------------------------|-------------------------------------|
| `dataset_tektonik.csv` | Historical tectonic earthquake data |
| `dataset_vulkanik.csv` | Volcanic activity data for tsunami  |

These datasets were used to train respective models, stored in the `models/` directory.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙋‍♂️ Contact

Maintainer: **Hendi Firmansyah**  
📧 Email: [hendi.firmansyah@bmkg.go.id](mailto:hendi.firmansyah@bmkg.go.id)

---

🌐 *Contributions, issue reports, and forks are welcome! Together, let's improve tsunami early warning systems through open science.*
