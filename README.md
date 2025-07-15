# Employee Resignation Prediction Using Behavioral Telemetry 

##  Project Goal
To predict which employees are likely to resign using behavioral signals like overtime, satisfaction, and promotions. This aligns with JPMorgan Chase's **Process Intelligence** strategy to analyze human-system interactions in digital workplaces.

##  Dataset
- 100,000 employee records
- Features: Demographics, work habits, performance, promotions, salary, satisfaction
- Target: `Resigned` (Boolean)

##  Methods Applied

### ✔ Feature Engineering
- `Tenure_Months`, `Projects_per_Year`, `Overtime_per_Week`
- `Engagement_Index` = avg of performance, satisfaction, promotions

### ✔ Data Balancing
- **SMOTE** oversampling of minority class (`Resigned=True`)
- Handled severe imbalance (90:10) effectively

### ✔ Models Tried
| Model | Result | Why |
|-------|--------|-----|
| **RandomForest + SMOTE + GridSearchCV** | Accuracy ~90%, Recall (True) ~0% | Couldn't learn from minority class |
| **XGBoost + SMOTE + Threshold=0.25** | Accuracy ~64%, Recall (True) ~32% | ✅ Better balance, caught 1 in 3 resignations |

### ✔ Explainability
- Used **SHAP** to identify key drivers:
  - `Overtime_per_Week`, `Engagement_Index`, `Satisfaction_Score`, `Promotions`

### ✔ Threshold Tuning
- Adjusted classification threshold from `0.5 → 0.25` to improve minority recall  
- Business logic: Better to flag a few false positives than miss resignations

---

##  Results Summary

| Metric | Value |
|--------|-------|
| Accuracy | 64% |
| Precision (`True`) | 10% |
| Recall (`True`) | 32% |
| F1-score (`True`) | 15% |

---

##  Outputs
- ✅ `resignation_predictor_model.pkl` — saved XGBoost model
- ✅ `high_risk_employees.csv` — top 100 employees likely to resign
- ✅ SHAP summary plots — for explainability

---

##  What I Learned
- SMOTE works, but **threshold tuning** makes or breaks minority-class models.
- Explainability is **critical** for business buy-in.
- Random Forest can miss patterns without domain-specific feature crafting.

---

##  Future Work
- Integrate real-time behavioral telemetry (clicks, task switch, calendar load)
- Build A/B testing simulator for HR interventions
- Add NLP sentiment analysis from exit interviews

---

##  Author
**Adarsh Vijayakrishnan**  
[LinkedIn](https://www.linkedin.com/in/adarshvijay08)

---

##  License
MIT


