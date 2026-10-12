# 🚀 YouTube Video Engagement Category Prediction
**An End-to-End Predictive Analytics & Machine Learning Pipeline Following the 7-Stage KDD (Knowledge Discovery in Databases) Lifecycle**

---

## 📌 Table of Contents
1. [Project Background & Business Understanding](#1-project-background--business-understanding)
2. [Data Acquisition & Dataset Architecture](#2-data-acquisition--dataset-architecture)
3. [The 7-Stage KDD Lifecycle Implementation](#3-the-7-stage-kdd-lifecycle-implementation)
4. [Feature Engineering & Preprocessing Breakdown](#4-feature-engineering--preprocessing-breakdown)
5. [Supervised Machine Learning Models](#5-supervised-machine-learning-models)
6. [Comprehensive Model Evaluation & Performance Matrix](#6-comprehensive-model-evaluation--performance-matrix)
7. [Results Visualization & Confusion Matrices](#7-results-visualization--confusion-matrices)
8. [Live Inference & Practical Application](#8-live-inference--practical-application)
9. [Future Scope & Enhancements](#9-future-scope--enhancements)

---

## 1. Project Background & Business Understanding
* **The Core Problem:** Content creators, digital marketers, and media networks face extreme uncertainty prior to publishing a video. Will a newly produced video achieve viral attention (`High Engagement`), or will it stall under high competition (`Low/Medium Engagement`)? 
* **The Solution:** By leveraging pre-publication metadata (such as title length, keyword optimization via tags, category classification, and publishing schedules), this system predicts engagement tiers before a video goes live. This allows creators to proactively optimize titles, refine tags, and select peak traffic publishing windows to maximize algorithmic distribution.

---

## 2. Data Acquisition & Dataset Architecture
* **Primary Source:** Regional trending YouTube statistics datasets (`INvideosDataset1.csv` and `USvideos_Dataset2.csv`) comprising over 37,000 raw video records.
* **Target Definition (`high_engagement`):** 
  * The raw target variable is absolute view counts (`views`).
  * To frame this as a robust binary classification problem, the median view threshold of the dataset is computed (e.g., **304,586 views** for the Indian regional dataset).
  * Videos with views **strictly greater than** the median are designated as **High Engagement (Class `1`)**.
  * Videos with views **at or below** the median are designated as **Low/Medium Engagement (Class `0`)**.

---

## 3. The 7-Stage KDD Lifecycle Implementation
This project rigorously follows the Knowledge Discovery in Databases (KDD) framework:
1. **Business Understanding:** Identifying content creator uncertainty and defining predictive objectives.
2. **Data Acquisition:** Ingesting multi-regional structured CSV records containing video telemetry.
3. **Data Understanding (EDA):** Visualizing class distributions and feature correlation structures.
4. **Data Cleaning:** Handling malformed text entries, null values, and parsing timestamp anomalies.
5. **Feature Engineering:** Extracting text lengths, tag quantities, and publishing temporal attributes.
6. **Modeling:** Training and validating Logistic Regression, Naive Bayes, and Random Forest classifiers.
7. **Evaluation:** Analyzing Precision, Recall, F1-Scores, Accuracy, and Confusion Matrices.

---

## 4. Feature Engineering & Preprocessing Breakdown
Raw metadata attributes were processed into high-signal numerical predictors:
* **`title_length`:** Character count of the video title (`len(title)`), measuring the impact of title conciseness and hook strength.
* **`tag_count`:** Total number of search keywords attached to a video, parsed by splitting pipe delimiters (`|`). A robust tag count indicates advanced SEO positioning.
* **`publish_hour` & `publish_day`:** Extracted directly from ISO 8601 timestamps (`publish_time`) to isolate peak user browsing hours (e.g., late afternoons and evenings) and days of the week.
* **Categorical Encoding:** `category_id` attributes mapped to continuous numerical indices via scikit-learn's `LabelEncoder`.
* **Feature Normalization:** All predictors scaled using `StandardScaler` to prevent feature magnitude dominance.
* **Data Splitting:** 80:20 stratified train-test split (`stratify=y`) ensuring balanced class representation in both training and test partitions.

---

## 5. Supervised Machine Learning Models
Three distinct algorithms were implemented to evaluate linear versus non-linear learning capabilities:
1. **Logistic Regression:** Serves as the linear baseline, optimizing decision boundaries through weighted coefficients.
2. **Gaussian Naive Bayes:** A fast probabilistic classifier relying on Bayes' theorem, assuming strict feature independence.
3. **Random Forest Classifier (Ensemble):** An ensemble framework combining 100 randomized decision trees. It captures non-linear synergies between title lengths, tag densities, and publish hours without requiring feature independence assumptions.

---

## 6. Comprehensive Model Evaluation & Performance Matrix
The models were evaluated on an unseen test set of 7,471 records. The **Random Forest Classifier** achieved superior performance across all evaluation metrics:

| Classifier Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 82% | 0.91 | 0.72 | 0.80 |
| **Gaussian Naive Bayes** | 73% | 0.97 | 0.49 | 0.65 |
| **Random Forest (Best)** | **93%** | **0.92** | **0.93** | **0.93** |

---

## 7. Results Visualization & Confusion Matrices
* **Comparative Bar Charts:** Side-by-side tracking of Accuracy, Precision, Recall, and F1-Scores clearly highlights Random Forest's dominance across metrics.
* **Confusion Matrix Heatmaps:** Generated individually for Logistic Regression, Naive Bayes, and Random Forest to visualize true positives, false positives, true negatives, and false negatives.

---

## 8. Live Inference & Practical Application
The notebook includes an interactive text-parsing inference cell. Content creators can input proposed video details:
```python
Title = "Top 10 Secret Features You Didn't Know Your Smartphone Had in 2026!"
Category_ID = 28
Tags_separated_by_pipe = "smartphone|tech|features|review|2026|gadgets|unboxing"
Publish_Time = "2026-10-15 19:00:00"
