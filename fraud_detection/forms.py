import pandas as pd
from django import forms

class PredictionForm(forms.Form):
    category = forms.IntegerField(
        label='Transaction Category',
    )
    amt = forms.DecimalField(
        label='Transaction Amount',
        max_digits=10,
        decimal_places=6,
        # help_text="Enter the transaction amount."
    )
    gender = forms.ChoiceField(
        label='Gender',
        choices=[(1, 'Male'), (0, 'Female')],
        widget=forms.RadioSelect,
        # help_text="Select the gender."
    )
    city_pop = forms.IntegerField(
        label='City Population',
        # help_text="Enter the population of the city where the transaction took place."
    )
    age = forms.DecimalField(
        label='Age',
        max_digits=10,
        decimal_places=6,
        # help_text="Enter the age of the person."
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
            'category': [int(self.cleaned_data['category'])],
            'amt': [float(self.cleaned_data['amt'])],
            'gender': [int(self.cleaned_data['gender'])],
            'city_pop': [int(self.cleaned_data['city_pop'])],
            'age': [float(self.cleaned_data['age'])],
            'trans_year': [int(self.cleaned_data['trans_year'])],
            'trans_month': [int(self.cleaned_data['trans_month'])],
            'trans_day': [int(self.cleaned_data['trans_day'])],
            'trans_hour': [int(self.cleaned_data['trans_hour'])],
            'distance_to_merch': [float(self.cleaned_data['distance_to_merch'])],
        }

        # Create the DataFrame with the correct data types
        df = pd.DataFrame(data)

        return df

