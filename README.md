# Exam Score Prediction and Analysis

This repository is a notebook-based exploration of the **Exam Score Prediction** dataset. It examines how study habits and student characteristics relate to exam scores, compares regression and classification models, and demonstrates a reusable preprocessing and model-selection workflow with scikit-learn.

The project is intended for interactive analysis and learning. It does not currently provide a command-line application, web service, or Python package.

## Project contents

| Path | Description |
| --- | --- |
| [`main.ipynb`](./main.ipynb) | Exploratory data analysis, feature preparation, exam-score regression, pass/fail classification, and experiments with model tuning and PCA. |
| [`training.ipynb`](./training.ipynb) | A more structured classification workflow with custom preprocessing transformers, stratified train/validation/test splits, grid search, final evaluation, and model serialization. |
| [`model/`](./model/) | Serialized scikit-learn model and grid-search examples. |
| [`requirements.txt`](./requirements.txt) | Python packages used by the notebooks. |
| [`LICENCE`](./LICENCE) | MIT license. |

## Dataset

The notebooks download the dataset through [KaggleHub](https://github.com/Kaggle/kagglehub) using the dataset identifier `kundanbedmutha/exam-score-prediction-dataset`. The expected CSV file is `Exam_Score_Prediction.csv`.

The dataset is not included in this repository. On the first notebook run, KaggleHub downloads it to its local cache. The notebooks expect the dataset to include fields used in the analysis, including `exam_score`, `study_hours`, `class_attendance`, `sleep_hours`, `sleep_quality`, `exam_difficulty`, and `gender`.

`main.ipynb` removes `course`, `study_method`, `facility_rating`, and `student_id` before preparing its numeric feature table. It also maps categorical values such as gender, sleep quality, internet access, and exam difficulty to numeric values.

## Getting started

Use Python with `pip` and Jupyter installed or available through the project dependencies.

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/javoxir-sv/exscore-dataset-analysis.git
   cd exscore-dataset-analysis
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Start JupyterLab and open either notebook:

   ```bash
   jupyter lab
   ```

Run notebook cells from top to bottom. The first data-loading cell in each notebook downloads the dataset with KaggleHub. If the dataset download is unavailable, check KaggleHub's current access requirements and make sure the dataset can be accessed from your environment.

## Notebook walkthrough

### `main.ipynb`: exploration and baseline models

This notebook is the broader exploratory analysis. It:

- Downloads and inspects the source CSV.
- Cleans and encodes selected columns for analysis.
- Visualizes score distributions and relationships using correlation heatmaps, box plots, histograms, scatter plots, KDE plots, and other charts.
- Predicts the numeric `exam_score` using linear regression, polynomial features, Ridge, Lasso, and a random-forest regressor.
- Defines a binary `pass` label (`exam_score >= 60`) and explores classifiers such as logistic regression and K-nearest neighbors.
- Demonstrates parameter search, model serialization with pickle/joblib, and a PCA-based classification pipeline.

The regression section uses `study_hours`, `class_attendance`, `sleep_hours`, and `sleep_quality` as its initial predictors. The notebook contains several independent experiments, so their setup and results should be interpreted within the relevant section rather than as one single benchmark.

### `training.ipynb`: structured pass/fail classification

This notebook presents a more organized workflow:

1. Creates a `status` label from `exam_score`: scores of 60 or higher are `"pass"`; lower scores are `"fail"`.
2. Uses a feature-extraction transformer to select study hours, class attendance, sleep hours, sleep quality, exam difficulty, and gender.
3. One-hot encodes categorical inputs while keeping the target separate.
4. Splits the data into training, validation, and test subsets using stratification and a fixed random seed.
5. Compares support vector machine, decision tree, and random forest classifiers with `GridSearchCV`.
6. Evaluates a classifier on the held-out test data and saves an estimator using joblib.

The notebook includes example validation and test scores in its markdown cells. Those values are illustrative; actual results can vary with the dataset version, dependency versions, and notebook execution.

## Model artifacts

The checked-in `model/` directory contains serialized scikit-learn examples:

- `grid2.pkl` and `grid2_joblib.pkl` are pickle and joblib serializations of a grid-search model from `main.ipynb`.
- `svc_824.sav` is a joblib-saved SVC estimator from the workflow in `training.ipynb`.

These files are not a complete inference application. In particular, the preprocessing steps and expected input schema are not packaged together with every serialized estimator. To make predictions reliably, use the same feature selection and encoding used during training, and ensure that the input columns and order match the estimator's training data.

Only load pickle/joblib files from sources you trust. These formats can execute code during deserialization, and scikit-learn does not guarantee compatibility of serialized estimators across arbitrary library versions.

## Reproducibility and interpretation

- The two notebooks use different targets and modeling approaches: numeric score prediction and binary pass/fail classification. Their results are not directly comparable.
- `training.ipynb` uses a fixed seed for its stratified data splits; `main.ipynb` includes splits without a fixed `random_state`, so rerunning those experiments can change the results.
- Grid search can take several minutes, particularly when it evaluates many parameter combinations. The training notebook sets `n_jobs=-1` to use available CPU cores.
- In the training notebook, the finalization example constructs a fixed SVC estimator rather than automatically passing the classifier returned by the grid-search selection step to `Finalize`. Review that cell if the intention is to score and save the best grid-searched classifier.
- Correlations and model scores describe patterns in this dataset; they do not establish causality or guarantee performance on other student populations.

## Dependencies

The project dependencies are listed in [`requirements.txt`](./requirements.txt): KaggleHub, Jupyter, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, SciPy, and Polars. The notebooks also use joblib and pickle for model serialization.

## License

This project is distributed under the MIT License. See [`LICENCE`](./LICENCE) for the full text.
