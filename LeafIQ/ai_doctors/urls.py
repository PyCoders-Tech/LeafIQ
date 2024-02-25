from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .api_views.views import TreeTypeViewSet

# router = DefaultRouter()
# router.register('tree-type', TreeTypeViewSet, basename='tree-type')
urlpatterns = [
    #api
    path("create-tree-type/", TreeTypeViewSet.as_view(), name="create-tree-type")
] #+ router.urls
