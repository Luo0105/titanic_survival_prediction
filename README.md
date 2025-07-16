# Titanic: Advanced Survival Prediction with Stacking Ensemble

This project tackles the classic Titanic survival prediction challenge on Kaggle. It demonstrates a complete, end-to-end machine learning workflow, employing advanced techniques for feature engineering, imputation, and model ensembling to achieve a robust and accurate prediction model.

**Project Notebook**: [`titanic-ver1.ipynb`](./titanic-ver1.ipynb)

---

### 🛠️ Tech Stack
* **Data Manipulation**: `Python`, `Pandas`, `Numpy`
* **Machine Learning**: `Scikit-learn`, `XGBoost`
* **Core Libraries Used**:
    * `KNNImputer` for sophisticated missing value imputation.
    * `StackingClassifier` for building a powerful ensemble model.
    * `GridSearchCV` for systematic hyperparameter tuning.

---

### 📈 Project Workflow

#### 1. Feature Engineering
A comprehensive feature engineering strategy was implemented to extract maximum value from the raw data:
* **Title Extraction**: Extracted passenger titles (e.g., 'Mr', 'Miss', 'Mrs') from the `Name` column. Rare titles were grouped into a single 'Rare' category to reduce noise.
* **Family Features**: Created `FamilySize` (by combining `SibSp` and `Parch`) and a binary `IsAlone` feature to capture family dynamics.
* **Categorical Encoding**: Used `LabelEncoder` for features like `Sex`, `Title`, and `Embarked`.

#### 2. Advanced Imputation
Missing values, particularly in the `Age` column, were handled with a sophisticated approach instead of simple statistical fills:
* **KNNImputer**: Utilized Scikit-learn's `KNNImputer` to predict missing `Age` values based on the most similar passengers (neighbors) using features like `Fare` and `Embarked`. This ensures that the imputed values are contextually relevant.
* **Standardization**: Features were scaled using `StandardScaler` before imputation to ensure the distance metric for KNN was not biased by feature magnitudes.

#### 3. Modeling: Stacking Ensemble
To leverage the strengths of multiple algorithms, a **Stacking Ensemble** model was constructed:
* **Base Models (Level 0 Estimators)**:
    1.  **XGBoost (`XGBClassifier`)**: A powerful gradient boosting model, excellent at capturing complex non-linear relationships.
    2.  **Random Forest (`RandomForestClassifier`)**: A robust bagging-based model, effective against overfitting.
    3.  **Logistic Regression (`LogisticRegression`)**: A stable linear model, providing a different "perspective".
* **Meta-Model (Level 1 Estimator)**:
    * A `LogisticRegression` model was used as the `final_estimator`. It takes the predictions from the base models as input and makes the final survival prediction, effectively learning how to best combine the base learners' outputs.

#### 4. Hyperparameter Tuning & Evaluation
* **GridSearchCV**: A `GridSearchCV` was applied to the entire `StackingClassifier` pipeline to systematically search for the optimal combination of hyperparameters for both the XGBoost and Random Forest base models.
* **Cross-Validation**: A 5-fold cross-validation (`cv=5`) strategy was used throughout the grid search. This provides a reliable estimate of the model's performance on unseen data and helps prevent overfitting to the training set.

---

### 🏁 Results & Key Learnings

* **Best CV Score**: The optimized stacking model achieved a cross-validation accuracy of **`0.8115184232000502`** on the training data.
* **Submission**: The final model was used to generate predictions on the test set, resulting in the submission file `submission_stacking.csv`.

#### Key Learnings:
1.  **Power of Ensembling**: This project clearly demonstrated that `StackingClassifier` can outperform individual models by combining their diverse strengths. It's a powerful technique for boosting performance.
2.  **Sophisticated Imputation Matters**: Using `KNNImputer` over simple mean/median imputation provides a more robust way to handle missing data, which is crucial for model accuracy.
3.  **Systematic Optimization**: `GridSearchCV` is an indispensable tool for tuning complex pipelines like stacking, ensuring that the final model is not just powerful in theory but also optimized in practice.
