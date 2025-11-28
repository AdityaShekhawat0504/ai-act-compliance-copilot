# **AI Act Compliance Copilot**

**AI Act Compliance Copilot** is a technical prototype demonstrating how organisations can assess, audit, monitor, and document AI systems under the upcoming EU AI Act.

It provides:

* A structured risk classification questionnaire
* A model audit framework (performance, fairness, explainability)
* A monitoring dashboard for tracking model performance and fairness over time
* A governance assistant powered by an LLM (OpenAI GPT-4o-mini)
* Automated PDF reporting

This project is intended for educational, research, and demonstration purposes.

---

## **Features**

### **1. Risk Classification**

A multi-section questionnaire covering:

* Domain and use-case
* Data characteristics and sensitivity
* Model characteristics
* Deployment impact
* Governance and oversight controls

The engine produces:

* A numerical risk score
* A risk level (low, limited, high, very high)
* A pillar-by-pillar breakdown
* Text-based justifications

The scoring logic takes inspiration from concepts in the EU AI Act (e.g., Annex III high-risk areas, governance controls).

---

### **2. Model Audit**

The platform provides three audit modes:

#### **a) Synthetic Credit Risk Demo**

* Generates a synthetic credit risk dataset
* Trains a Random Forest classifier
* Computes accuracy, ROC AUC, and classification report
* Computes Statistical Parity Difference (SPD) for fairness
* Provides SHAP-based explainability
* Logs every run into a monitoring file

Useful for demonstrating the full audit flow.

---

#### **b) Upload Your Own Dataset (Train Inside the App)**

* Upload a CSV
* Select the target column and optional protected attribute
* The system trains a Random Forest model
* Computes performance and fairness metrics
* Logs the audit result
* Useful for quickly analysing custom tabular data

---

#### **c) Upload a Pre-Trained Model (Black-Box Audit)**

* Upload a `.pkl` or `.joblib` model (any scikit-learn compatible model)
* Upload a test dataset with ground-truth labels
* The system:

  * Loads the model
  * Extracts numeric features
  * Computes performance metrics
  * Computes SPD fairness metrics
  * Logs the audit run

This mode allows auditing of **existing production models** without retraining.

---

### **3. Monitoring Dashboard**

A simple CSV-backed monitoring log powers a dashboard that displays:

* All audit entries in tabular form
* Filtering by model name and domain
* Performance trends over time (accuracy, ROC AUC)
* Fairness trends (SPD)
* Distribution of risk levels

This simulates a lightweight compliance-focused MLOps monitoring system.

---

### **4. Governance Copilot**

A governance assistant powered by an LLM (OpenAI GPT-4o-mini).
It can:

* Explain risk classification outputs
* Interpret fairness metrics
* Summarise audit results
* Answer natural-language questions about the model
* Draft concise documentation sections for compliance or reporting

The assistant uses session context plus the latest audit results to produce tailored explanations.

---

### **5. Reporting**

The system can generate a structured PDF report combining:

* Risk classification results
* Model performance metrics
* Fairness metrics
* Summary explanations

This is useful for internal reviews, presentations, and documentation drafts.

---

## **Project Structure**

```
ai-act-compliance-copilot/
├── demo/
│   ├── app.py                   # Main Streamlit application
│   ├── data_utils.py            # Synthetic dataset generation
│   ├── explanations.py          # SHAP explainability utilities
│   ├── fairness_utils.py        # Fairness metrics (SPD)
│   ├── model_training.py        # ML training and evaluation helpers
│   ├── monitoring_utils.py      # Audit logging and loading
│   ├── report_generator.py      # PDF report generation
│   ├── risk_engine.py           # Rule-based risk scoring engine
│   ├── requirements.txt         # Python dependencies
│   └── (runtime artifacts: shap_summary.png, monitoring_log.csv)
├── .gitignore
└── README.md
```

---

## **Getting Started**

### **1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/ai-act-compliance-copilot.git
cd ai-act-compliance-copilot/demo
```

### **2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

(Windows PowerShell)

```powershell
.\.venv\Scripts\activate
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set your OpenAI API key**

```bash
export OPENAI_API_KEY="your_key_here"
```

To make it persistent (WSL/Linux):

```bash
echo 'export OPENAI_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### **5. Run the application**

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## **Intended Use**

This project was developed as a portfolio and demonstration tool focusing on:

* AI governance
* EU AI Act compliance
* Fairness and explainability
* Monitoring and documentation

It is **not** intended as a full legal compliance solution and should not be used as a substitute for professional assessment.

---

## **Roadmap**

Potential future improvements include:

* Additional fairness metrics (equal opportunity, AOD, disparate impact)
* Drift detection (data drift, prediction drift)
* Model card generation
* Support for XGBoost/LightGBM pipelines
* Persistent database-backed monitoring
* Integration with enterprise IAM or MLOps systems

---




