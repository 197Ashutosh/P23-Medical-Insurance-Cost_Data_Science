# 🏥 Medical Insurance Cost Prediction using Linear Regression

This project builds a predictive model to estimate **medical insurance charges** based on patient details like age, BMI, smoking status, and region. The model is trained using **Linear Regression** on a cleaned dataset of 2772 records, with in-depth **exploratory data analysis (EDA)**, **outlier handling**, and an **interactive terminal-based prediction system**.

---

## 📁 Dataset Summary

- 📄 Source: `medical_insurance.csv`  
- 🔢 Rows: 2772  
- 🎯 Target: `charges` (medical cost in USD)

| Feature     | Description                            |
|-------------|----------------------------------------|
| `age`       | Age of the individual (18–64)          |
| `sex`       | Gender (`male`, `female`)              |
| `bmi`       | Body Mass Index                        |
| `children`  | Number of children covered             |
| `smoker`    | Smoking status (`yes`, `no`)           |
| `region`    | Region in the US                       |
| `charges`   | Insurance charges (USD)                |

---

## 🔍 Exploratory Data Analysis (EDA)

The following visualizations were used:

- 📈 Distribution plots for numeric variables
- 📊 Count plots for categorical features (`sex`, `smoker`, `region`)
- 🟩 Correlation heatmap
- 📉 Age vs Charges (highlighted by smoking status)
- 🧍 BMI vs Charges (highlighted by smoker)
- 🌍 Average charges by region, gender, children

### 🧠 Key Insights:

- Smokers and individuals with higher BMI incur higher charges.
- Age has a moderate positive correlation with charges.
- Region and sex show weaker effects on charge amounts.

---

## 🧼 Outlier Detection & Handling

- ✅ Used **IQR method** to cap extreme BMI values
- ✅ Applied **Z-score method** to identify high `charges`
- ✅ Used `np.log1p()` transformation on `charges` to reduce skewness

---

## 🤖 Model: Linear Regression

After encoding categorical variables using `pd.get_dummies()`:

```python
model = LinearRegression()
model.fit(X_train, y_train)
