import pandas as pd
import joblib
from sklearn.metrics import classification_report


# Function to load the model and make predictions
def make_predictions(input_file, model_path):
    # Load the trained pipeline model
    model = joblib.load(model_path)

    # Load new data for prediction
    new_data = pd.read_csv(input_file)

    # Make predictions using the pre-trained model pipeline
    predictions = model.predict(new_data)

    # Print predictions
    print("Predictions for the new data:")
    print(predictions)

    # Optionally, you can save predictions to a CSV file
    output_df = new_data.copy()
    output_df['Predicted_AdoptionSpeed'] = predictions
    output_df.to_csv('../data/predictions_output.csv', index=False)
    print("Predictions saved to 'data/predictions_output.csv'")

    print(classification_report(new_data['AdoptionSpeed'], predictions))


if __name__ == "__main__":
    # Provide the path to your input file (new_input.csv)
    input_file = 'data/new_input.csv'
    model_path = 'models/xgboost_model_pipeline.pkl'
    make_predictions(input_file, model_path)
