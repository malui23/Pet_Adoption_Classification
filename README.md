
# Pet Adoption Classification

This project focuses on building a machine learning pipeline to classify pet adoption speed. Using structured data about pets (e.g., age, breed, color, health, etc.), the goal is to predict how quickly a pet will be adopted. The pipeline preprocesses the data, handles categorical and numerical features, performs feature selection, and trains an XGBoost classifier to achieve accurate predictions.
## Key Features

- **Preprocessing**: Handles missing values, encodes categorical data, and scales numerical features.
- **Class Imbalance Handling**: To address class imbalance, *SMOTE (Synthetic Minority Over-sampling Technique)* is applied, generating synthetic data points for underrepresented classes. This technique improves the model's ability to generalize across all classes.
- **Class Merging**: Some classes within the target variable, AdoptionSpeed, are merged to simplify the classification problem. This helps in improving model performance by reducing the number of rare classes.
- **Target Encoding**: Encodes *breed* features based on the target variable for better performance.
- **Feature Selection**: Uses *SelectFromModel* with *XGBoost* to reduce dimensionality.
- **Model Training**: Implements a robust XGBoost classifier for classification.
- **Evaluation**: Provides performance metrics including a *classification report, ROC-AUC curve and confusion matrix.*
- **Pipeline Automation**: Combines all steps into a single Pipeline for streamlined processing and inference.
## Project Structure

- **data/**: Contains datasets used for training and testing the model.

    - **new_input.csv**: Raw data intended for preprocessing and training.
    - **train.csv**: Preprocessed data used for classification.

- **pipeline/**: Contains scripts for building and training the machine learning pipelines.

    - **train_xgb.py**: Script that preprocesses data, trains the pipeline using XGBClassifier, and evaluates the model.
- **models/**: Stores trained machine learning models.

    - **xgboost_model_pipeline.pkl**: Saved XGBoost model pipeline for future predictions.
- **notebooks/**: contains Jupyter Notebook files that document different stages of the project.

    - **data_understanding.ipynb**: This notebook focuses on the initial analysis and understanding of the data, providing insights into its structure, basic statistics and visualization.
    - **feature_engineering.ipynb**: This file explores the feature engineering process (existing features are transformed to improve model performance), *filling the NaN-vales, encoding categorical data and scaling*.
    - **other_models_eval.ipynb**: This notebook evaluates alternative models (*LogisticRegression, RandomForest and AdaBoost*) and compares their performance. 
    - **xgb_eval.ipynb**: This notebook specifically evaluates the *XGBoost model*.
**In the *other_models_eval.ipynb* and *xgb_eval.ipynb* files, I used a slightly different version of the dataset, which is processed using SMOTE (Synthetic Minority Over-sampling Technique) and class merging to address class imbalances. *(train.csv already oversampled and its classes also merged)*.**

- **predict.py**: is used to make predictions on new data using a pre-trained XGBoost model pipeline. It reads data from the file new_input.csv, processes it through the same pipeline that was used during training, and outputs the predicted results.

- **requirements.txt**: Lists all Python dependencies required for running the project.

- **README.md**: Contains a description of the project, its purpose, and instructions for usage.

- **venv/**: Virtual environment to ensure dependency isolation for the project.





## How to Use

- Install Dependencies: Install required Python packages using requirements.txt. 
- Run the Pipeline: Execute train_xgb.py to preprocess data, train the model on train.csv data, and evaluate performance.
- Save the Model: Trained models are saved as .pkl files for future use. 
- Data: Use the raw data (new_input.csv) as the input dataset, pipeline includes preprocessing new data.