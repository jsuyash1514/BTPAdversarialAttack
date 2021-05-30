from django.db import models


class Result(models.Model):
    active = models.BooleanField(default=False)
    input = models.ImageField(upload_to='media/results')
    output1 = models.CharField(null=True, blank=True, max_length=400)
    output2 = models.CharField(null=True, blank=True, max_length=400)

    def __str__(self):
        return f'{self.id}: active:{self.active}'
