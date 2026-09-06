# Term Deposit Subscriber Predictor

A Streamlit web app that predicts whether a bank customer is likely to subscribe to a **term deposit**, based on a machine learning model trained on the **Bank Marketing dataset (`bank-full.csv`)**.

🔗 **Live app:** https://termdepositsubscriberpredictor.streamlit.app/
📦 **Repository:** https://github.com/arya-arun-123/Term_Deposit_Prediction_App

---

## 📌 Overview

Banks run direct marketing campaigns (mostly phone calls) to convince clients to subscribe to a term deposit. This project uses historical campaign data to build a classification model that predicts the likelihood of a customer saying **"yes"** to a term deposit offer — helping marketing teams prioritize the customers most worth contacting.

The app takes in customer and campaign details through a simple form and returns a prediction: **will subscribe** or **will not subscribe**.

---

## 📊 Dataset

- **File:** `bank-full.csv`
- **Source:** UCI Bank Marketing Dataset
- **Size:** ~45,211 rows, 17 columns (before preprocessing)
- **Target variable:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)
- **Class balance:** Imbalanced (far more "no" than "yes" responses), so accuracy alone isn't a reliable metric — precision, recall, and F1-score were also evaluated.

### Key features used
| Feature | Description |
|---|---|
| `age` | Age of the client |
| `job` | Type of job |
| `marital` | Marital status |
| `education` | Education level |
| `balance` | Average yearly account balance |
| `housing` | Has a housing loan? |
| `loan` | Has a personal loan? |
| `contact` | Contact communication type |
| `month` | Last contact month of the year |
| `campaign` | Number of contacts during this campaign |
| `pdays` | Days since last contacted from a previous campaign |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of the previous marketing campaign |

**Columns dropped** during preprocessing:
- `duration` — not known before a call is placed, and directly leaks the outcome (if duration = 0, target is always "no")
- `default` — highly imbalanced, added little predictive value
- `day` — day of the month was found to have negligible effect on the target

---

## ⚙️ Data Preprocessing

1. **Missing value & duplicate handling** — duplicate rows removed, dataset checked for nulls.
2. **Outlier analysis** — performed on skewed numerical columns.
3. **Encoding**
   - `LabelEncoder` applied to binary columns: `housing`, `loan`, `y`
   - `OneHotEncoder` applied to nominal categorical columns: `job`, `marital`, `education`, `contact`, `month`, `poutcome`
4. **Feature relevance check** — mutual information scores were calculated between each feature and the target; all features were retained.
5. **Scaling** — numerical features scaled using `MinMaxScaler` / `StandardScaler`.
6. **Train-test split** — 80/20 split, `random_state=42`.

---

## 🤖 Model Building & Selection

Several classification algorithms were trained and compared, including:

- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Bagging & AdaBoost ensembles (applied to multiple base models)
- Artificial Neural Network (ANN)

Each model was evaluated on **Accuracy, Precision, Recall, and F1-score**, with extra weight given to F1 and Recall due to the class imbalance in the dataset.

### Best performing models

| Criterion | Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Best Accuracy | Decision Tree (Boosting) | 0.717 | 0.669 | 0.195 | 0.302 |
| Best F1-score | **Bagging SVM** | 0.643 | 0.448 | 0.588 | **0.509** |
| Best Recall | **Bagging SVM** | 0.643 | 0.448 | **0.588** | 0.509 |

**Final model deployed: Bagging SVM**, selected for its balanced precision/recall trade-off and the best F1-score — a more meaningful metric than raw accuracy on this imbalanced dataset, where correctly catching potential subscribers (recall) matters.

The trained model was serialized with `pickle` (`best_model.pkl`) and loaded into the Streamlit app for real-time inference.

---

## 🖥️ How the App Works

The app walks the user through a **3-step guided form**, each step must be verified (checkbox) before moving to the next:

1. **👤 Step 1 — Customer Profile**: age, job/occupation, marital status, education level.
2. **💰 Step 2 — Financial Info**: average yearly balance, housing loan (yes/no), personal loan (yes/no), credit default (yes/no).
3. **📞 Step 3 — Campaign & Predict**: contact type, last contact month, number of contacts this campaign, days since last contact (`pdays`), number of previous contacts, outcome of the previous campaign — then a **Predict** button triggers inference.

Behind the scenes, on clicking predict:
1. Inputs are assembled into numerical/binary and categorical DataFrames.
2. The saved **`encoder.pkl`** (OneHotEncoder) transforms the categorical fields (`job`, `marital`, `education`, `contact`, `month`, `poutcome`).
3. Encoded + numeric/binary features are combined and reindexed to match the exact feature order the model was trained on.
4. The saved **`scaler.pkl`** scales the combined feature vector.
5. The saved **`model.pkl`** (Bagging SVM) predicts the class and, where available, the subscription probability.
6. The result is shown as a styled success card (🎯 likely to subscribe) or danger card (❌ unlikely to subscribe) along with the predicted probability.

The UI uses a custom dark, emerald-green theme built with injected CSS for high contrast/readability.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — web app framework & UI (tabbed navigation, custom dark theme)
- **Pandas / NumPy** — data handling
- **Scikit-learn** — preprocessing (encoding, scaling), model building, evaluation
- **Joblib** — loading serialized model artifacts (`encoder.pkl`, `scaler.pkl`, `model.pkl`)
- **TensorFlow / Keras** — ANN experimentation (in the training notebook)
- **Matplotlib / Seaborn** — exploratory data analysis (in the training notebook)

**`requirements.txt`:**
```
streamlit>=1.32
pandas
numpy
scikit-learn
joblib
```

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/arya-arun-123/Term_Deposit_Prediction_App.git
cd Term_Deposit_Prediction_App

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app expects the trained artifacts (`encoder.pkl`, `scaler.pkl`, `model.pkl`) to be present inside the `artifacts/` folder — it will raise an error and stop if any of them are missing.

---

## 📁 Project Structure

```
Term_Deposit_Prediction_App/
├── .streamlit/                      # Streamlit config (theme/settings)
├── artifacts/                       # Serialized model artifacts
│   ├── encoder.pkl                  # OneHotEncoder for categorical features
│   ├── scaler.pkl                   # Feature scaler
│   └── model.pkl                    # Trained Bagging SVM classifier
├── Bank_Management (3).ipynb        # EDA, preprocessing & model training notebook
├── app.py                           # Streamlit application (UI + inference)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md
```

---

## 📈 Future Improvements

- Address class imbalance further (e.g., SMOTE, class-weighting) to improve recall.
- Add feature importance / SHAP explanations to the app for interpretability.
- Expand hyperparameter tuning across more ensemble configurations.
- Add batch prediction support (CSV upload) alongside single-record input.

---

## ⚠️ Disclaimer

As noted in the app itself: predictions are **not guaranteed to be accurate** — the model can make mistakes and should not be used as the sole basis for real business decisions.

---

## 📄 License

Released under the **MIT License**. Built on the publicly available UCI Bank Marketing dataset for educational/demonstration purposes.
