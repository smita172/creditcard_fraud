from django import forms

class PredictionForm(forms.Form):
    trans_date_trans_time = forms.DateTimeField(
        label='Transaction Date & Time (YYYY-MM-DD HH:MM:SS)',
        widget=forms.TextInput(attrs={'placeholder': '2023-01-01 12:00:00'})
    )
    cc_num = forms.CharField(
        label='Credit Card Number',
        max_length=16,
        widget=forms.TextInput(attrs={'placeholder': '1234567890123456'})
    )
    merchant = forms.CharField(
        label='Merchant',
        max_length=50,
        widget=forms.TextInput(attrs={'placeholder': 'Merchant Name'})
    )
    category = forms.CharField(
        label='Category',
        max_length=20,
        widget=forms.TextInput(attrs={'placeholder': 'Category (e.g., shopping, food)'})
    )
    amt = forms.FloatField(
        label='Transaction Amount',
        widget=forms.NumberInput(attrs={'placeholder': 'e.g., 100.00'})
    )
    city_pop = forms.IntegerField(
        label='City Population',
        widget=forms.NumberInput(attrs={'placeholder': 'e.g., 50000'})
    )
    job = forms.CharField(
        label='Job',
        max_length=50,
        widget=forms.TextInput(attrs={'placeholder': 'e.g., engineer, teacher'})
    )
    dob = forms.DateField(
        label='Date of Birth (YYYY-MM-DD)',
        widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'})
    )
    merch_lat = forms.FloatField(
        label='Merchant Latitude',
        widget=forms.NumberInput(attrs={'placeholder': 'e.g., 34.0522'})
    )
    merch_long = forms.FloatField(
        label='Merchant Longitude',
        widget=forms.NumberInput(attrs={'placeholder': '-118.2437'})
    )
    # Add more fields as necessary based on the features your model needs
