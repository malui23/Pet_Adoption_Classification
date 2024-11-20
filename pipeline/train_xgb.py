import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
random_state = 13

if __name__ == '__main__':
    # Завантажуємо дані
    df = pd.read_csv('../data/train.csv')

    # Видаляємо непотрібні колонки та створюємо копію для роботи, щоб не змінювати оригінальний датафрейм
    df.drop(['Unnamed: 0', 'Name', 'RescuerID', 'PetID', 'Dewormed', 'Description'], axis=1, inplace=True)

    # Вибір ознак (X) та цільової змінної (y)
    X = df.drop('AdoptionSpeed', axis=1)
    y = df['AdoptionSpeed']

    # Розподіляємо на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state,)

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    cat_num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])
    breed_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', TargetEncoder(handle_unknown='ignore'))
    ])

    num_features = ['Age', 'Fee', 'Quantity', 'VideoAmt', 'PhotoAmt']
    cat_features = ['ColorName_x', 'ColorName_y', 'ColorName', 'StateName_x']
    cat_num_features = ['Type', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health']
    breed_features = ['BreedName_x', 'BreedName_y']

    preprocessor = ColumnTransformer(transformers=[
        ('numeric', num_transformer, num_features),
        ('categorical', cat_transformer, cat_features),
        ('cat_num', cat_num_transformer, cat_num_features),
        ('breed', breed_transformer, breed_features)
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(
            xgb.XGBClassifier(eta=0.1, max_depth=10, subsample=0.5, alpha=5, reg_lambda=10, scale_pos_weight=1),
            threshold=0.01)),
        ('model', xgb.XGBClassifier(eta=0.1, max_depth=10, subsample=0.5, alpha=5, reg_lambda=10, scale_pos_weight=1))
    ])

    # Тренуємо модель на тренувальних даних
    pipeline.fit(X_train, y_train)

    # Передбачаємо на тестових даних
    y_pred = pipeline.predict(X_test)

    # Оцінка якості моделі
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Якщо ви хочете зберегти модель для подальшого використання:
    import joblib

    joblib.dump(pipeline, 'xgboost_model_pipeline.pkl')
