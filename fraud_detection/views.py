from django.shortcuts import render
from .forms import PredictionForm
from .utils import model
# from .utils import  predictions_df

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
