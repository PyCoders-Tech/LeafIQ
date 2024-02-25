from django.contrib import admin
from .models import TreeType

@admin.register(TreeType)
class TreeTypeAdmin(admin.ModelAdmin):
    pass

