from django import forms

class PredictionForm(forms.Form):
    category = forms.CharField(
        label='Transaction Category',
        max_length=50
    )
    amt = forms.DecimalField(
        label='Transaction Amount',
        max_digits=10,  # total number of digits allowed
        decimal_places=7  # number of decimal places
    )
    city_pop = forms.IntegerField(
        label='City Population'
    )
    age = forms.DecimalField(
        label='Age',
        max_digits=7,
        decimal_places=6
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
        decimal_places=7
    )
    gender_M = forms.ChoiceField(
        label='Gender',
        choices=[(1, 'Male'), (0, 'Female')],
        widget=forms.RadioSelect
    )
