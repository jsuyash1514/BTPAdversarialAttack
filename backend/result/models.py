from django.db import models


class Result(models.Model):
    active = models.BooleanField(default=False)
    input = models.ImageField(upload_to='media/results')
    output1 = models.ImageField(
        upload_to='media/results', null=True, blank=True)
    output2 = models.ImageField(
        upload_to='media/results', null=True, blank=True)

    def __str__(self):
        return f'{self.id}: active:{self.active}'
