# Random_forest.ipynb: Hyperparameter Tuning of Random Forest Regressor

## 📋 Project Description
This Jupyter notebook, `Random_forest.ipynb`, implements and optimizes a **Random Forest Regressor** model, likely for a regression task associated with the Pima Indians **Diabetes dataset**.

The project focuses on:
* Loading and initial exploratory analysis of the dataset.
* Handling missing data by replacing placeholder zeros with NaN values.
* Training a **Random Forest Regressor** model.
* Using **`GridSearchCV`** for exhaustive search-based **hyperparameter tuning** to find the optimal model configuration.
* Evaluating the performance of both the base and optimized models using key regression metrics.

---

## 💾 Data & Preprocessing
### Dataset
The notebook requires the file **`diabetes.csv`** to be present in the working directory. Based on the columns targeted for cleaning, the project uses a dataset related to diabetes prediction/regression, such as the Pima Indians Diabetes dataset.

### Missing Value Handling
Zero values (`0`) in the following critical features are treated as missing data and replaced with **`np.nan`**:
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`

### Visualization
The notebook includes a step to generate **histograms** (`df.hist(bins=50, figsize=(20,15))`) for visualizing the distribution of features after initial cleaning.

---

## ⚙️ Methodology
### Model
The core machine learning model used is the **RandomForestRegressor** from scikit-learn.

### Hyperparameter Tuning (GridSearchCV)
To find the best combination of parameters for the Random Forest model, the notebook utilizes **`GridSearchCV`**.

| Hyperparameter | Values Explored |
| :--- | :--- |
| `n_estimators` (Number of trees) | `[100, 200, 300]` |
| `max_depth` (Max depth of trees) | `[None, 10, 20]` |
| `min_samples_split` (Min samples to split a node) | `[2, 5, 10]` |
| `min_samples_leaf` (Min samples in a leaf node) | `[1, 2, 4]` |

The grid search uses **5-fold cross-validation (`cv=5`)** and is optimized using the **`neg_mean_squared_error`** scoring metric.

### Evaluation Metrics
The performance of the models (both initial and optimized) is assessed using standard regression metrics:
* **Mean Absolute Error (MAE)**
* **R-squared ($R^2$) Score**

---

## 💻 Dependencies
To run this notebook, you need a Python environment with the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
