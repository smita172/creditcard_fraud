import pandas as pd
from django import forms

class PredictionForm(forms.Form):
    amt = forms.DecimalField(
        label='Transaction Amount',
        max_digits=10,
        decimal_places=6,
        # help_text="Enter the transaction amount."
    )
    category = forms.ChoiceField(
        label='Transaction Category',
        choices=[
            ('0','entertainment'),
            ('1','food_dining'),
            ('2','gas_transport'),
            ('3','grocery_net'),
            ('4','grocery_pos'),
            ('5','health_fitness'),
            ('6','home'),
            ('7','kids_pets'),
            ('8','misc_net'),
            ('9','misc_pos'),
            ('10','personal_care'),
            ('11','shopping_net'),
            ('12','shopping_pos'),
            ('13','travel'),
        ]
    )
    lat = forms.DecimalField(
        label='Customer Latitude',
        max_digits=10,
        decimal_places=6,
    )
    long = forms.DecimalField(
        label='Customer Longitude',
        max_digits=10,
        decimal_places=6,
    )
    merch_lat = forms.DecimalField(
        label='Merchant Latitude',
        max_digits=10,
        decimal_places=6,
    )
    merch_long = forms.DecimalField(
        label='Merchant Longitude',
        max_digits=10,
        decimal_places=6,
    )
    # gender = forms.ChoiceField(
    #     label='Gender',
    #     choices=[(1, 'Male'), (0, 'Female')],
    #     widget=forms.RadioSelect(attrs={'class': 'radio-group'}),
    #     # help_text="Select the gender."
    # )
    city_pop = forms.IntegerField(
        label='City Population',
        # help_text="Enter the population of the city where the transaction took place."
    )
    age = forms.IntegerField(
        label='Age',
    )
    trans_year = forms.IntegerField(
        label='Transaction Year'
    )
    trans_month = forms.IntegerField(
        label='Transaction Month'
    )
    trans_day = forms.IntegerField(
        label='Transaction Day'
    )
    trans_hour = forms.IntegerField(
        label='Transaction Hour'
    )
    distance_to_merch = forms.DecimalField(
        label='Distance to Merchant (km)',
        max_digits=10,
        decimal_places=6,
        # help_text="Enter the distance in kilometers."
    )


    def to_dataframe(self):
        """Converts form data to a pandas DataFrame with correct data types."""
        data = {
            'amt': [float(self.cleaned_data['amt'])],
            'category': [int(self.cleaned_data['category'])],
            # 'gender': [int(self.cleaned_data['gender'])],
            'lat': [float(self.cleaned_data['lat'])],
            'long': [float(self.cleaned_data['long'])],
            'merch_lat': [float(self.cleaned_data['merch_lat'])],
            'merch_long': [float(self.cleaned_data['merch_long'])],
            'city_pop': [int(self.cleaned_data['city_pop'])],
            'age': [int(self.cleaned_data['age'])],
            'trans_year': [int(self.cleaned_data['trans_year'])],
            'trans_month': [int(self.cleaned_data['trans_month'])],
            'trans_day': [int(self.cleaned_data['trans_day'])],
            'trans_hour': [int(self.cleaned_data['trans_hour'])],
            'distance_to_merch': [float(self.cleaned_data['distance_to_merch'])],
        }

        # Create the DataFrame with the correct data types
        df = pd.DataFrame(data)

        return df

class CSVUploadForm(forms.Form):
    file = forms.FileField(
        label='Upload CSV',
        help_text='Upload a CSV file with the required columns.'
    )

