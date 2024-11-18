# csv_app/models.py
from django.db import models

class ProcessedData(models.Model):
    category = models.IntegerField()
    amt = models.DecimalField(max_digits=10, decimal_places=6)
    gender = models.IntegerField()  # 1 for Male, 0 for Female
    city_pop = models.IntegerField()
    age = models.DecimalField(max_digits=10, decimal_places=6)
    trans_year = models.IntegerField()
    trans_month = models.IntegerField()
    trans_day = models.IntegerField()
    trans_hour = models.IntegerField()
    distance_to_merch = models.DecimalField(max_digits=10, decimal_places=6)
    result = models.CharField(max_length=255)

    def process_row(self):
        # Example processing logic for each row
        self.result = f"Processed transaction: {self.category}, amount: {self.amt}"
        self.save()
