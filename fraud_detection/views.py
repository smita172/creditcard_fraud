from django.shortcuts import render, redirect
from .forms import PredictionForm
from .utils import model
import csv
import io
from django.contrib import messages
from .forms import CSVUploadForm
from .models import ProcessedData
from django.http import HttpResponse
import pickle
# from .utils import  predictions_df
import pandas as pd
def home_view(request):
    return render(request, 'home.html')

# def input_prediction(request):
#     prediction_result = None
#     if request.method == 'POST':
#         form = PredictionForm(request.POST)
#         if form.is_valid():
#             # Convert form data to a DataFrame using the to_dataframe method
#             input_df = form.to_dataframe()
#
#             # Predict using the loaded model
#             print(input_df)
#             prediction = model.predict(input_df)
#             print(prediction)
#             prediction_result = prediction[0]  # Get single prediction
#             return render(request, 'input_result.html', {'prediction_result': prediction_result})
#
#     else:
#         form = PredictionForm()
#
#     return render(request, 'input_prediction.html', {
#         'form': form,
#         'prediction_result': prediction_result,
#     })


def input_prediction(request):
    """Handles the transaction input form, processes predictions, and returns the result view."""

    # Initialize variables
    prediction_result = None
    transaction_data = None  # To store cleaned transaction details

    if request.method == 'POST':
        form = PredictionForm(request.POST)

        if form.is_valid():
            # Capture transaction data for display on result page
            transaction_data = form.cleaned_data

            # Convert form data to DataFrame and filter only expected columns
            input_df = form.to_dataframe()
            expected_columns = [
                'category', 'amt', 'gender', 'city_pop', 'age',
                'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'distance_to_merch'
            ]

            # Drop any columns not in the expected list
            input_df = input_df.reindex(columns=expected_columns)

            # Check if input_df has any missing columns after reindexing
            missing_columns = [col for col in expected_columns if col not in input_df.columns]
            if missing_columns:
                return render(request, 'input_prediction.html', {
                    'form': form,
                    'error_message': f"Missing required columns: {', '.join(missing_columns)}",
                })

            # Predict using the model
            try:
                prediction = model.predict(input_df)
                prediction_result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
            except Exception as e:
                prediction_result = f"Error in prediction: {str(e)}"

            # Render the result template with prediction result and transaction details
            return render(request, 'input_result.html', {
                'prediction_result': prediction_result,
                'transaction_data': transaction_data,
            })
    else:
        form = PredictionForm()

    # Render the input form page
    return render(request, 'input_prediction.html', {
        'form': form,
        'prediction_result': prediction_result,
    })


REQUIRED_COLUMNS = [
    'category', 'amt', 'gender', 'city_pop', 'age',
    'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'distance_to_merch'
]

def safe_convert(value, target_type):
    """Converts a value to a target_type, returning 'invalid data' if conversion fails."""
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return "invalid data"


def upload_csv(request):
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['file']
            decoded_file = csv_file.read().decode('utf-8').splitlines()
            reader = csv.DictReader(decoded_file)

            # Check for required columns
            if not set(REQUIRED_COLUMNS).issubset(reader.fieldnames):
                messages.error(request, f"CSV must contain columns: {', '.join(REQUIRED_COLUMNS)}")
                return redirect('upload_csv')

            # Prepare a buffer to hold the new CSV with results
            output = io.StringIO()
            writer = csv.writer(output)

            # Write the header with an extra column for the result
            header = reader.fieldnames + ['prediction']
            writer.writerow(header)

            for row in reader:
                # Convert the row to the format needed for prediction, using safe conversion
                features = pd.DataFrame([{
                    'category': safe_convert(row['category'], int),
                    'amt': safe_convert(row['amt'], float),
                    'gender': safe_convert(row['gender'], int),
                    'city_pop': safe_convert(row['city_pop'], float),
                    'age': safe_convert(row['age'], float),
                    'trans_year': safe_convert(row['trans_year'], int),
                    'trans_month': safe_convert(row['trans_month'], int),
                    'trans_day': safe_convert(row['trans_day'], int),
                    'trans_hour': safe_convert(row['trans_hour'], int),
                    'distance_to_merch': safe_convert(row['distance_to_merch'], float)
                }])

                # Check if any feature is marked as 'invalid data'
                if features.isin(['invalid data']).any().any():
                    prediction = "invalid data"
                else:
                    # Predict using the loaded model
                    try:
                        with open(r'C:\Users\rethek\Desktop\Windsor\Fall 2024\project\model.pkl', 'rb') as model_file2:
                            model2 = pickle.load(model_file2)
                        prediction_result = model2.predict(features)[0]
                        prediction = "fraud" if prediction_result == 1 else "not fraud"
                    except AttributeError:
                        messages.error(request, "Model prediction failed. Please check model compatibility.")
                        return redirect('upload_csv')

                # Append the prediction to the row data
                row['prediction'] = prediction
                writer.writerow([row[column] for column in header])

            # Create a response for file download
            response = HttpResponse(output.getvalue(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="processed_results_with_predictions.csv"'
            return response

    else:
        form = CSVUploadForm()

    return render(request, 'csv_prediction.html', {'form': form})