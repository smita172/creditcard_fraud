import pandas as pd
from django.shortcuts import render
from .forms import PredictionForm
from .utils import model, predictions_df

def predict(request):
    prediction_result = None
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract data from form
            input_data = {field: form.cleaned_data[field] for field in form.cleaned_data}
            input_df = pd.DataFrame([input_data])

            # Predict using the loaded model
            prediction = model.predict(input_df)
            prediction_result = prediction[0]  # Get single prediction

    else:
        form = PredictionForm()

    return render(request, 'predict.html', {
        'form': form,
        'prediction_result': prediction_result,
    })
