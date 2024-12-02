import json
from django.shortcuts import render, redirect
from .forms import PredictionForm
import csv
import io
from django.contrib import messages
from .forms import CSVUploadForm
from django.http import HttpResponse
from django.conf import settings
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
from .utils import get_coordinates, haversine

REQUIRED_COLUMNS = [
    'amt','category','lat', 'long','merch_lat','merch_long', 'city_pop', 'age',
    'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'distance_to_merch'
]

CATEGORIES = [
    {'id': 0, 'description': 'entertainment'},
    {'id': 1, 'description': 'food_dining'},
    {'id': 2, 'description': 'gas_transport'},
    {'id': 3, 'description': 'grocery_net'},
    {'id': 4, 'description': 'grocery_pos'},
    {'id': 5, 'description': 'health_fitness'},
    {'id': 6, 'description': 'home'},
    {'id': 7, 'description': 'kids_pets'},
    {'id': 8, 'description': 'misc_net'},
    {'id': 9, 'description': 'misc_pos'},
    {'id': 10, 'description': 'personal_care'},
    {'id': 11, 'description': 'shopping_net'},
    {'id': 12, 'description': 'shopping_pos'},
    {'id': 13, 'description': 'travel'},
]

def safe_convert(value, target_type):
    """Converts a value to a target_type, returning 'invalid data' if conversion fails."""
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return "invalid data"
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

            # Extract customer and merchant city names
            # customer_city = transaction_data.get('customer_city')
            # merchant_city = transaction_data.get('merchant_city')

            # try:
            #     # Fetch coordinates for both cities
            #     cust_lat, cust_lon = get_coordinates(customer_city)
            #     merch_lat, merch_lon = get_coordinates(merchant_city)
            #
            #     # Calculate distance using Haversine formula
            #     distance_to_merch = haversine(cust_lat, cust_lon, merch_lat, merch_lon)
            #
            #     # Add calculated fields to the transaction data
            #     transaction_data['lat'] = cust_lat
            #     transaction_data['long'] = cust_lon
            #     transaction_data['merch_lat'] = merch_lat
            #     transaction_data['merch_long'] = merch_lon
            #     transaction_data['distance_to_merch'] = distance_to_merch
            #
            # except ValueError as e:
            #     # Handle invalid city names
            #     form.add_error(None, f"Error: {str(e)}")
            #     return render(request, 'input_prediction.html', {'form': form})
            # except Exception as e:
            #     # Handle API or other errors
            #     messages.error(request, f"An error occurred: {str(e)}")
            #     return redirect('input_prediction')
            #
            #     # Convert form data to DataFrame and filter only expected columns
            # input_df = form.to_dataframe()
            # expected_columns = [
            #     'amt', 'category', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 'age',
            #     'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'distance_to_merch'
            # ]
            #
            # # Add the dynamically calculated values to the DataFrame
            # input_df['lat'] = cust_lat
            # input_df['long'] = cust_lon
            # input_df['merch_lat'] = merch_lat
            # input_df['merch_long'] = merch_lon
            # input_df['distance_to_merch'] = distance_to_merch

            # Convert form data to DataFrame and filter only expected columns
            input_df = form.to_dataframe()
            expected_columns = [
                'amt', 'category', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 'age',
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

            try:
                # with open('finalmodel.pkl', 'rb') as model_file:
                #     model = pickle.load(model_file)
                model = joblib.load(r'finalmodel.pkl')
                prediction = model.predict(input_df)
                prediction_result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
            except Exception as e:
                print(str(e))
                messages.error(request, f"Prediction failed: {str(e)}")
                return redirect('input_prediction')

            return render(request, 'input_result.html', {
                'prediction_result': prediction_result,
                'transaction_data': transaction_data,
            })
    else:
        form = PredictionForm()

    # Render the input form page
    return render(request, 'input_prediction.html', {
        'form': form,
    })


def upload_csv(request):
    predictions = []
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




            model2 = joblib.load(r'finalmodel.pkl')

            for row in reader:
                features = pd.DataFrame([{
                    col: safe_convert(row[col],
                                      int if col in ['category', 'city_pop','age', 'trans_year', 'trans_month', 'trans_day',
                                                     'trans_hour'] else float)
                    for col in REQUIRED_COLUMNS
                }])
                fraud_status = "Invalid Data" if features.isnull().values.any() else (
                    "Fraudulent" if model2.predict(features)[0] == 1 else "Not Fraudulent"
                )
                row['Fraud Prediction (Fraudulent or Not Fraudulent)'] = fraud_status
                writer.writerow(row.values())
                predictions.append(row)

            response = HttpResponse(output.getvalue(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="processed_results_with_predictions.csv"'
            return response

    else:
        form = CSVUploadForm()

    return render(request, 'csv_prediction.html', {
        'form': form,
        'categories': CATEGORIES,
    })

def csv_visualization(request):
    try:
        csv_file_path = os.path.join(settings.BASE_DIR, 'processed_results_with_predictions.csv')

        # Check if the file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError("CSV file not found.")

        # Load the CSV data
        data = pd.read_csv(csv_file_path)

        # 1. Fraud vs Not Fraud
        # Calculate fraud vs not fraud counts
        fraud_counts = data['prediction'].value_counts()

        # Check if there are any transactions to prevent division by zero
        total_transactions = fraud_counts.sum()
        if total_transactions == 0:
            fraud_data = {
                "labels": ["No Transactions"],
                "data": [100],  # Default to 100% for "No Transactions"
            }
        else:
            # Normalize to percentages
            fraud_data = {
                "labels": fraud_counts.index.tolist(),
                "data": [(count / total_transactions) * 100 for count in fraud_counts.values],
                # Normalize to percentage
            }


        fraudulent_transactions = data[data['prediction'] == 'Fraudulent']
        non_fraudulent_transactions = data[data['prediction'] == 'Not Fraudulent']
        if not fraudulent_transactions.empty:
            amt_fraudulent_distribution = fraudulent_transactions['amt'].value_counts(bins=5, sort=False)
            amt_fraudulent_data = {
                "labels": [f"{int(bin.left)}-{int(bin.right)}" for bin in amt_fraudulent_distribution.index],
                "data": [(count / amt_fraudulent_distribution.sum()) * 100 for count in
                         amt_fraudulent_distribution.values],
            }
        else:
            amt_fraudulent_data = {
                "labels": ["No Fraudulent Transactions"],
                "data": [100],  # Placeholder for an empty chart
            }

            # Handle Non-Fraudulent transactions
        if not non_fraudulent_transactions.empty:
            amt_non_fraudulent_distribution = non_fraudulent_transactions['amt'].value_counts(bins=5, sort=False)
            amt_non_fraudulent_data = {
                "labels": [f"{int(bin.left)}-{int(bin.right)}" for bin in amt_non_fraudulent_distribution.index],
                "data": [(count / amt_non_fraudulent_distribution.sum()) * 100 for count in
                         amt_non_fraudulent_distribution.values],
            }
        else:
            amt_non_fraudulent_data = {
                "labels": ["No Non-Fraudulent Transactions"],
                "data": [100],  # Placeholder for an empty chart
            }




        # 3. Transaction Volume by Month
        trans_month_counts = data['trans_month'].value_counts().sort_index()
        month_data = {
            "labels": trans_month_counts.index.tolist(),
            "data": trans_month_counts.values.tolist(),
        }

        # 4. Number of Fraud Transactions by Category
        fraud_transactions = data[data['prediction'] == 'Fraudulent']
        if not fraud_transactions.empty:
            fraud_counts_by_category = fraud_transactions['category'].value_counts()
        else:
            fraud_counts_by_category = pd.Series([0], index=["No Fraudulent Transactions"])

        category_fraud_data = {
            "labels": fraud_counts_by_category.index.tolist(),
            "data": fraud_counts_by_category.values.tolist(),
        }

        # 5. Distance to Merchant Distribution
        dist_distribution = data['distance_to_merch'].value_counts(bins=10, sort=False)
        distance_data = {
            "labels": [f"{round(bin.left, 1)}-{round(bin.right, 1)}" for bin in dist_distribution.index],
            "data": dist_distribution.values.tolist(),
        }

        dist_fraudulent_transactions = data[data['prediction'] == 'Fraudulent']

        if not dist_fraudulent_transactions.empty:
            # Calculate the distribution of 'distance_to_merch' for fraudulent transactions
            fraudulent_dist_distribution = dist_fraudulent_transactions['distance_to_merch'].value_counts(bins=10,
                                                                                                     sort=False)
            fraud_distance_data = {
                "labels": [f"{round(bin.left, 1)}-{round(bin.right, 1)}" for bin in fraudulent_dist_distribution.index],
                "data": fraudulent_dist_distribution.values.tolist(),
            }
        else:
            # Handle case where there are no fraudulent transactions
            fraud_distance_data = {
                "labels": ["No Transactions"],
                "data": [100],  # Default value
            }

        # 6. Age Distribution
        age_distribution = data['age'].value_counts(bins=10, sort=False)
        age_data = {
            "labels": [f"{int(bin.left)}-{int(bin.right)}" for bin in age_distribution.index],
            "data": age_distribution.values.tolist(),
        }

        # Filter fraudulent transactions
        fraudulent_age_data = data[data['prediction'] == 'Fraudulent']

        # Check if there are any fraudulent transactions
        if not fraudulent_age_data.empty:
            # Calculate age distribution for fraudulent transactions
            age_bins_fraud = fraudulent_age_data['age'].value_counts(bins=10, sort=False)
            fraudulent_age_chart_data = {
                "labels": [f"{int(bin.left)}-{int(bin.right)}" for bin in age_bins_fraud.index],
                "data": age_bins_fraud.values.tolist(),  # Use raw counts for each bin
            }
        else:
            # Handle the case where no fraudulent transactions exist
            fraudulent_age_chart_data = {
                "labels": ["No Fraudulent Transactions"],
                "data": [0],  # Placeholder for no data
            }

        # Prepare context for rendering
        context = {
            'fraud_data': json.dumps(fraud_data),
            'fraudulent_age_chart_data':json.dumps(fraudulent_age_chart_data),
            # 'amt_data': json.dumps(amt_data),
            'amt_fraudulent_data': json.dumps(amt_fraudulent_data),
            'amt_non_fraudulent_data': json.dumps(amt_non_fraudulent_data),
            # 'amt_non_fraudulent_data': json.dumps(amt_non_fraudulent_data),
            'month_data': json.dumps(month_data),
            'category_fraud_data': json.dumps(category_fraud_data),
            'distance_data': json.dumps(distance_data),
            'fraud_distance_data': json.dumps(fraud_distance_data),
            'age_data': json.dumps(age_data),
            'categories': CATEGORIES,
        }
        return render(request, 'csv_visualization.html', context)

    except FileNotFoundError as e:
        return render(request, 'csv_visualization.html', {'error_message': str(e)})

    except Exception as e:
        return render(request, 'csv_visualization.html', {'error_message': 'An unexpected error occurred: ' + str(e)})


# def csv_visualization(request):
#     try:
#         csv_file_path = os.path.join(settings.BASE_DIR, 'processed_results_with_predictions.csv')
#
#         # Check if the file exists
#         if not os.path.exists(csv_file_path):
#             raise FileNotFoundError("CSV file not found.")
#
#     # csv_file_path = os.path.join(settings.BASE_DIR, 'processed_results_with_predictions.csv')
#         data = pd.read_csv(csv_file_path)
#
#         # print(data['prediction'].value_counts())
#     # 1. Fraud vs Not Fraud
#         fraud_counts = data['prediction'].value_counts()
#         fraud_plot_path = os.path.join(settings.MEDIA_ROOT, 'fraud_vs_not_fraud.png')
#         plt.figure(figsize=(6, 4))
#         fraud_counts.plot(kind='bar', color=['green', 'red'])
#         plt.title('Fraud vs Not Fraud')
#         plt.xlabel('Prediction')
#         plt.ylabel('Count')
#         plt.xticks(rotation=0)
#         plt.savefig(fraud_plot_path)
#         plt.close()
#
#         # 2. Transaction Amount Distribution
#         amt_dist_plot_path = os.path.join(settings.MEDIA_ROOT, 'amt_distribution.png')
#         plt.figure(figsize=(6, 4))
#         data['amt'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
#         plt.title('Transaction Amount Distribution')
#         plt.xlabel('Transaction Amount')
#         plt.ylabel('Frequency')
#         plt.savefig(amt_dist_plot_path)
#         plt.close()
#
#         # 3. Gender-wise Fraud Analysis
#         # gender_fraud_counts = data.groupby('gender')['prediction'].value_counts().unstack()
#         # gender_fraud_plot_path = os.path.join(settings.MEDIA_ROOT, 'gender_fraud.png')
#         # gender_fraud_counts.plot(kind='bar', figsize=(6, 4), stacked=True, color=['green', 'red'])
#         # plt.title('Gender-wise Fraud Analysis')
#         # plt.xlabel('Gender (0: Female, 1: Male)')
#         # plt.ylabel('Count')
#         # plt.xticks(rotation=0)
#         # plt.savefig(gender_fraud_plot_path)
#         # plt.close()
#
#         # 4. Transaction Volume by Month
#         trans_month_counts = data['trans_month'].value_counts().sort_index()
#         month_plot_path = os.path.join(settings.MEDIA_ROOT, 'trans_by_month.png')
#         plt.figure(figsize=(6, 4))
#         trans_month_counts.plot(kind='bar', color='purple')
#         plt.title('Transaction Volume by Month')
#         plt.xlabel('Month')
#         plt.ylabel('Number of Transactions')
#         plt.savefig(month_plot_path)
#         plt.close()
#
#         # Number of Fraud Transactions by Category
#         fraud_transactions = data[data['prediction'] == 'Fraudulent']  # Filter for fraud transactions
#
#         # Check if there are any fraudulent transactions
#         if not fraud_transactions.empty:
#             fraud_counts_by_category = fraud_transactions['category'].value_counts()
#         else:
#             # If no fraudulent transactions, create an empty DataFrame for plotting
#             fraud_counts_by_category = pd.Series([0], index=["NFT"])
#
#         # Save the fraud count plot for categories
#         category_fraud_plot_path = os.path.join(settings.MEDIA_ROOT, 'fraud_transactions_by_category.png')
#         plt.figure(figsize=(6, 4))
#         fraud_counts_by_category.plot(kind='bar', color='orange')
#         plt.title('Number of Fraud Transactions by Category')
#         plt.xlabel('Category')
#         plt.ylabel('Number of Fraud Transactions')
#         plt.savefig(category_fraud_plot_path)
#         plt.close()
#
#         # 6. Distance to Merchant Distribution
#         dist_plot_path = os.path.join(settings.MEDIA_ROOT, 'distance_distribution.png')
#         plt.figure(figsize=(6, 4))
#         data['distance_to_merch'].plot(kind='hist', bins=20, color='cyan', edgecolor='black')
#         plt.title('Distance to Merchant Distribution')
#         plt.xlabel('Distance')
#         plt.ylabel('Frequency')
#         plt.savefig(dist_plot_path)
#         plt.close()
#
#         # 7. Age Distribution
#         age_plot_path = os.path.join(settings.MEDIA_ROOT, 'age_distribution.png')
#         plt.figure(figsize=(6, 4))
#         data['age'].plot(kind='hist', bins=20, color='pink', edgecolor='black')
#         plt.title('Age Distribution')
#         plt.xlabel('Age')
#         plt.ylabel('Frequency')
#         plt.savefig(age_plot_path)
#         plt.close()
#
#         # Prepare context for rendering
#         context = {
#             'fraud_plot_url': settings.MEDIA_URL + 'fraud_vs_not_fraud.png',
#             'amt_dist_plot_url': settings.MEDIA_URL + 'amt_distribution.png',
#             # 'gender_fraud_plot_url': settings.MEDIA_URL + 'gender_fraud.png',
#             'month_plot_url': settings.MEDIA_URL + 'trans_by_month.png',
#             'category_fraud_plot_url': settings.MEDIA_URL + 'fraud_transactions_by_category.png',
#             'dist_plot_url': settings.MEDIA_URL + 'distance_distribution.png',
#             'age_plot_url': settings.MEDIA_URL + 'age_distribution.png',
#             'categories': CATEGORIES,
#         }
#         return render(request, 'csv_visualization.html', context)
#
#     except FileNotFoundError as e:
#         return render(request, 'csv_visualization.html', {'error_message': str(e)})
#
#     except Exception as e:
#         return render(request, 'csv_visualization.html', {'error_message': 'An unexpected error occurred: ' + str(e)})
#
