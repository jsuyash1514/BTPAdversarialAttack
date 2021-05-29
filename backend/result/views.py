from rest_framework.response import Response
from result.models import Result
from rest_framework import viewsets
from .serializers import ResultSerializer


class ResultViewset(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = Result.objects.filter(active=True).order_by('-id')
    serializer_class = ResultSerializer
