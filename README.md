# 2501PTDS_Classification_Project
# Lorian Barrett (Individual Project)
## Analysing News Articles Dataset

[![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](#4-environment-setup)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://2501ptdsclassificationproject-9vemkbuksitvq9ey8pnoxy.streamlit.app/)


<div id="main image" align="center">
  <img src="https://github.com/ereshia/2401FTDS_Classification_Project/blob/main/announcement-article-articles-copy-coverage.jpg" width="550" height="300" alt=""/>
</div>

---

## Table of Contents
* [1. Project Overview](#project-description)
* [2. Dataset](#dataset)
* [3. Packages](#packages)
* [4. Environment Setup](#environment)
  * [4.1 Conda Setup](#conda)
  * [4.2 venv Setup (Alternative)](#venv)
* [5. MLflow Experiment Tracking](#mlflow)
* [6. Streamlit Deployment](#streamlit)
* [7. Team Members](#team-members)
* [8. Troubleshooting](#troubleshooting)
* [9. Results Summary](#results-summary)
* [10. How to Run the App](#run-app)


---

## 1. Project Overview <a class="anchor" id="project-description"></a>

Your team has been hired as data science consultants for a news outlet to create classification models using Python and deploy them as a web application with Streamlit.  
The goal is to apply machine learning techniques to natural language processing (NLP) tasks — building, evaluating, and deploying models to classify news articles into topics.

This project covers the full data science workflow:
- Data loading and exploration  
- Text preprocessing and feature engineering  
- Model training and evaluation  
- Model tracking with MLflow  
- Deployment via Streamlit  

**Stakeholders:** Editorial team, IT support, management, and readers — all of whom benefit from improved content categorization and enhanced user experience.

---

## 2. Dataset <a class="anchor" id="dataset"></a>

The dataset contains news articles that need to be classified into one of the following categories:
`Business`, `Technology`, `Sports`, `Education`, and `Entertainment`.

You can find both `train.csv` and `test.csv` datasets [here](https://github.com/ereshia/2401FTDS_Classification_Project/tree/main/Data/processed).

| **Column** | **Description** |
|-------------|-----------------|
| **Headlines** | Title of the news article |
| **Description** | Short summary of the article |
| **Content** | Full text of the article |
| **URL** | Source link |
| **Category** | The article’s true label/category |

---

## 3. Packages <a class="anchor" id="packages"></a>

The following core dependencies are required to execute this project:

- `pandas 2.2.2`
- `numpy 1.26`
- `matplotlib 3.8.4`
- `scikit-learn`
- `streamlit`
- `mlflow`
- `jupyter`
- `nltk` (for tokenization, lemmatization, and stopword removal)
- `seaborn` (for confusion matrix visualization)
---

## 4. Environment Setup <a class="anchor" id="environment"></a>

It’s highly recommended to use a **virtual environment** to isolate dependencies.  
Below are two options — choose whichever best fits your setup.

> 💡 **Note for first-timers:**  
> The initial setup can take several minutes (sometimes up to 20–30 on slower or corporate networks).  
> This is normal — the environment manager needs to download and install large data science libraries for the first time.

---

### 4.1 Using Conda (Recommended for unrestricted setups) <a class="anchor" id="conda"></a>

If you have **Anaconda** or **Miniconda** installed, follow these steps:

#### 🧩 Create the environment
```bash
conda create --name nlp_project python=3.10
```

#### 🔹 Activate the environment
```bash
conda activate nlp_project
```

#### 🔹 Install project dependencies
```bash
conda install pip
pip install -r requirements.txt
```

#### 🔹 Add environment to Jupyter (optional)
```bash
python -m ipykernel install --user --name=nlp_project
```

#### 🔹 Deactivate the environment
```bash
conda deactivate
```

---

### 4.2 Using venv (Alternative for restricted or corporate setups) <a class="anchor" id="venv"></a>

> ⚠️ **Note:**  
> Some organizations restrict software downloads, blocking external installers like Anaconda/Miniconda.  
> In that case, use Python's built-in `venv` module to create a lightweight isolated environment.

#### 🧩 Create a virtual environment
Open your terminal or VS Code terminal and navigate to your project folder:
```bash
cd "C:\Users\lbarrett\@Workspace\DataScience Course\Module_7_NPL\NLP Project\2501PTDS_Classification_Project"
```

Then run:
```bash
python -m venv venv
```

This creates a folder called `venv` that contains your virtual environment.

#### 🔹 Activate the environment
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

#### 🔹 Install dependencies
If a `requirements.txt` file exists:
```bash
pip install -r requirements.txt
```

Otherwise, install manually:
```bash
pip install pandas numpy matplotlib scikit-learn streamlit mlflow jupyter nltk
```

#### 🔹 Download NLTK data (required for text processing)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"
```

#### 🔹 Add environment to Jupyter (optional)
```bash
python -m ipykernel install --user --name=venv
```

#### 🔹 Deactivate the environment
```bash
deactivate
```

✅ *You now have a fully functional environment — no external tools required.*

---

## 5. MLflow Experiment Tracking <a class="anchor" id="mlflow"></a>

**MLflow** is used to manage and track machine-learning experiments, parameters, metrics, and artifacts.  
It ensures **reproducibility** and provides a clean interface for comparing model runs.

### 🔹 What is Logged
- Model parameters – algorithm, TF-IDF configuration, dataset details  
- Performance metrics – training and validation accuracy  
- Artifacts – trained model and confusion-matrix plot (`confusion_matrix.png`)  
- (Optional) Input example – for automatic model-signature inference

### 🔹 Example MLflow Code
```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:///C:/Users/lbarrett/mlruns")

with mlflow.start_run(run_name="News_Classifier_Final"):
    mlflow.log_param("Model", "LogisticRegression")
    mlflow.log_param("Vectorizer", "TF-IDF (max_features=5000, ngram_range=(1,2))")
    mlflow.log_metric("Training Accuracy", float(train_accuracy))
    mlflow.log_metric("Validation Accuracy", float(test_accuracy))
    mlflow.sklearn.log_model(model, name="model")

    # Log confusion matrix image
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

print("✅ MLflow experiment and confusion matrix logged successfully!")


## 6. Streamlit <a class="anchor" id="streamlit"></a>

### What is Streamlit?

[Streamlit](https://www.streamlit.io/) is a Python-based framework for creating interactive web applications — perfect for deploying data science models with minimal effort.

> “Streamlit is the easiest way for data scientists to build beautiful, performant apps in only a few hours — all in pure Python.”

---

### 🔹 Description of files
| File Name | Description |
|------------|-------------|
| `base_app.py` | Streamlit app definition file |

---

### 🔹 Running the Streamlit app locally

Ensure the required packages are installed:
```bash
pip install -U streamlit numpy pandas scikit-learn
```

Then launch the app:
```bash
cd Streamlit
streamlit run base_app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

---

### 🔹 Deploying to Streamlit Cloud

1. Push your latest code to GitHub.  
2. Visit [Streamlit Cloud](https://share.streamlit.io/).  
3. Sign in using GitHub and click **New App → Select Repo**.  
4. Set `Streamlit/base_app.py` as the entry file.  

Your model will now be live and shareable via a public URL 🎉

---

## 7. Team Members <a class="anchor" id="team-members"></a>

| Name | Email |
|------|--------|
| [Lorian Barrett](https://github.com/lorianbarrett) | lorian.barrett@ninetyone.com |

> **Note:** This is an individual project completed by Lorian Barrett.  
> [Oludare Adekunle](https://github.com/DareSandtech) and [Claudia Elliot-Wilson] served as course conveners for guidance and oversight.

---

## 8. Troubleshooting <a class="anchor" id="troubleshooting"></a>

| Issue | Solution |
|-------|-----------|
| `conda` not recognized | Ensure Anaconda/Miniconda is installed and added to your PATH, or use the `venv` setup. |
| `venv` won’t activate in VS Code | Make sure your terminal shell is set to PowerShell or Command Prompt, not Git Bash. |
| Long installation time | This is normal for first-time setup — large libraries (e.g. pandas, scikit-learn) may take several minutes to download and compile. |
| Jupyter doesn’t detect kernel | Run `python -m ipykernel install --user --name=<env_name>` and restart VS Code. |
| Streamlit app not launching | Check that all required libraries are installed, then rerun `streamlit run base_app.py`. |

---


---

## 9. Results Summary <a class="anchor" id="results-summary"></a>

### 📈 Model Performance
The Logistic Regression classifier achieved **excellent overall accuracy** on the test dataset.

| Metric | Result |
|---------|---------|
| **Training Accuracy** | ~99 % |
| **Validation Accuracy** | ~97 % |
| **Best-performing Categories** | Education & Entertainment |
| **Minor Misclassifications** | Business ↔ Technology, Sports ↔ Entertainment |

### 🧩 Interpretation
- The **confusion matrix** confirms high model accuracy and strong category separation.  
- Minor confusion between Business and Technology is expected due to overlapping language and shared terminology.  
- MLflow tracking ensures reproducibility by logging parameters, metrics, and the confusion-matrix artifact for every run.

### 🚀 Deployment Impact
The final model was deployed with **Streamlit**, allowing users to:
- Input any news headline or article,  
- Instantly receive a **predicted category** and **confidence score**, and  
- Visualize category probabilities through a dynamic bar chart.

This completes an **end-to-end NLP project** — from raw data → model training → evaluation → interactive deployment.

---

---

## 10. How to Run the App <a class="anchor" id="run-app"></a>

Follow the steps below to activate your virtual environment and launch the Streamlit web application.

### 🪟 For Windows Users
1. Open **VS Code** or **Command Prompt**, then navigate to your project folder:
   ```bash
   cd "C:\Users\lbarrett\@Workspace\DataScience Course\Module_7_NPL\NLP Project\2501PTDS_Classification_Project"

2. Activate your virtual environment:
```bash
venv\Scripts\activate

(Your terminal prompt should now show (venv) — this means your environment is active.)

3. Run Streamlit app
```bash
streamlit run Streamlit/base_app.py


