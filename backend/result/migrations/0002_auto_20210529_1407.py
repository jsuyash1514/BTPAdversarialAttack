# Generated by Django 3.2.3 on 2021-05-29 08:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('result', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='result',
            old_name='output',
            new_name='output1',
        ),
        migrations.AddField(
            model_name='result',
            name='output2',
            field=models.ImageField(blank=True, null=True, upload_to='media/results'),
        ),
    ]