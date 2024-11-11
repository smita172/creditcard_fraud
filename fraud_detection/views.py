import pandas as pd
from django.shortcuts import render
from .forms import PredictionForm
from .utils import model
# from .utils import  predictions_df

def home_view(request):
    return render(request, 'home.html')

def input_prediction(request):
    prediction_result = None
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Convert form data to a DataFrame using the to_dataframe method
            input_df = form.to_dataframe()

            # Predict using the loaded model
            print(input_df)
            prediction = model.predict(input_df)
            print(prediction)
            prediction_result = prediction[0]  # Get single prediction

    else:
        form = PredictionForm()

    return render(request, 'input_prediction.html', {
        'form': form,
        'prediction_result': prediction_result,
    })
