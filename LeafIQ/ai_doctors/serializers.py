from rest_framework import serializers
from .models import TreeType

class TreeTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = TreeType
        fields = '__all__'