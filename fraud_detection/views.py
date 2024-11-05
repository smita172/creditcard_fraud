from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Helper function to create visualization
def create_visualization(data):
    plt.figure(figsize=(10, 5))
    plt.hist(data['amt'], bins=30, alpha=0.5, label='Amount')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return image_base64

def input_view(request):
    if request.method == 'POST':
        input_data = {
            'trans_date_trans_time': request.POST.get('trans_date_trans_time'),
            'cc_num': request.POST.get('cc_num'),
            'amt': float(request.POST.get('amt')),
            'merchant': request.POST.get('merchant'),
            'category': request.POST.get('category'),
        }
        is_fraud = False  # Placeholder for fraud detection result
        visualization = create_visualization(pd.DataFrame([input_data]))
        return render(request, 'input_result.html', {'result': is_fraud, 'visualization': visualization})
    return render(request, 'input_form.html')

def csv_upload_view(request):
    if request.method == 'POST':
        file = request.FILES['csv_file']
        data = pd.read_csv(file)
        predictions = [{'is_fraud': False, **row} for _, row in data.iterrows()]
        visualization = create_visualization(data)
        return render(request, 'csv_result.html', {'data': predictions, 'visualization': visualization})
    return render(request, 'csv_upload.html')
