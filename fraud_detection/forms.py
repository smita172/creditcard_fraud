import pandas as pd
from django import forms
from django.core.exceptions import ValidationError
from decimal import Decimal

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
        decimal_places=5,
    )
    long = forms.DecimalField(
        label='Customer Longitude',
        max_digits=10,
        decimal_places=5,
    )
    merch_lat = forms.DecimalField(
        label='Merchant Latitude',
        max_digits=10,
        decimal_places=5,
    )
    merch_long = forms.DecimalField(
        label='Merchant Longitude',
        max_digits=10,
        decimal_places=5,
    )
    # gender = forms.ChoiceField(
    #     label='Gender',
    #     choices=[(1, 'Male'), (0, 'Female')],
    #     widget=forms.RadioSelect(attrs={'class': 'radio-group'}),
    #     # help_text="Select the gender."
    # )

    # customer_city = forms.CharField(label='Customer City', max_length=100)
    # merchant_city = forms.CharField(label='Merchant City', max_length=100)
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
        decimal_places=5,
        # help_text="Enter the distance in kilometers."
    )

    def clean_city_pop(self):
        city_pop = self.cleaned_data.get('city_pop')
        if city_pop is not None and not isinstance(city_pop, int):
            raise ValidationError('City population must be an integer.')
        return city_pop

    def clean_trans_year(self):
        trans_year = self.cleaned_data.get('trans_year')
        if trans_year is not None and not isinstance(trans_year, int):
            raise ValidationError('Transaction year must be an integer.')
        return trans_year

    def clean_trans_month(self):
        trans_month = self.cleaned_data.get('trans_month')
        if trans_month is not None and (trans_month < 1 or trans_month > 12):
            raise ValidationError('Transaction month must be between 1 and 12.')
        return trans_month

    def clean_trans_day(self):
        trans_day = self.cleaned_data.get('trans_day')
        if trans_day is not None and (trans_day < 1 or trans_day > 31):
            raise ValidationError('Transaction day must be between 1 and 31.')
        return trans_day

    def clean_trans_hour(self):
        trans_hour = self.cleaned_data.get('trans_hour')
        if trans_hour is not None and (trans_hour < 1 or trans_hour > 24):
            raise ValidationError('Transaction hour must be between 1 and 24.')
        return trans_hour

    def clean_amt(self):
        amt = self.cleaned_data.get('amt')
        if amt is not None and not isinstance(amt, Decimal):
            raise ValidationError('Transaction amount must be an decimal.')
        return amt

    def clean_age(self):
        age = self.cleaned_data.get('age')
        if age is not None and not isinstance(age, int):
            raise ValidationError('Age must be an integer.')
        return age

    def clean_distance_to_merch(self):
        distance_to_merch = self.cleaned_data.get('distance_to_merch')
        if distance_to_merch is not None and not isinstance(distance_to_merch, Decimal):
            raise ValidationError('Distance of customer to merchant must be an decimal.')
        return distance_to_merch

    # def clean_lat(self):
    #     lat = self.cleaned_data.get('lat')
    #     if lat is None or not isinstance(lat, Decimal):
    #         raise ValidationError('Customer latitude must be a valid decimal value.')
    #     if lat < Decimal('-90') or lat > Decimal('90'):
    #         raise ValidationError('Customer latitude must be between -90 and 90.')
    #     return lat
    #
    # # Validate Customer Longitude
    # def clean_long(self):
    #     long = self.cleaned_data.get('long')
    #     if long is None or not isinstance(long, Decimal):
    #         raise ValidationError('Customer longitude must be a valid decimal value.')
    #     if long < Decimal('-180') or long > Decimal('180'):
    #         raise ValidationError('Customer longitude must be between -180 and 180.')
    #     return long

    # # Validate Merchant Latitude
    # def clean_merch_lat(self):
    #     merch_lat = self.cleaned_data.get('merch_lat')
    #     if merch_lat is None or not isinstance(merch_lat, Decimal):
    #         raise ValidationError('Merchant latitude must be a valid decimal value.')
    #     if merch_lat < Decimal('-90') or merch_lat > Decimal('90'):
    #         raise ValidationError('Merchant latitude must be between -90 and 90.')
    #     return merch_lat
    #
    # # Validate Merchant Longitude
    # def clean_merch_long(self):
    #     merch_long = self.cleaned_data.get('merch_long')
    #     if merch_long is None or not isinstance(merch_long, Decimal):
    #         raise ValidationError('Merchant longitude must be a valid decimal value.')
    #     if merch_long < Decimal('-180') or merch_long > Decimal('180'):
    #         raise ValidationError('Merchant longitude must be between -180 and 180.')
    #     return merch_long

    def clean(self):
        cleaned_data = super().clean()
        # Add form-level validations here (e.g., cross-field validation).
        return cleaned_data

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

