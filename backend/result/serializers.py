from result.models import *
from rest_framework import serializers

from dl.utils import f
from dl.utils import g


class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = '__all__'

    def create(self, validated_data):
        instance = super().create(validated_data)
        instance.output1 = f(instance.input.name)
        instance.output2 = g(instance.input.name)
        instance.active = True
        instance.save()
        return instance
