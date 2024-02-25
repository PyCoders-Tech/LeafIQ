from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from ..ML.cnn_models import predict_cnn_result
from ..models import TreeType
from ..serializers import TreeTypeSerializer

class TreeTypeViewSet(CreateAPIView):
    queryset = TreeType.objects.all()
    serializer_class = TreeTypeSerializer
    authentication_classes = [SessionAuthentication, BasicAuthentication]


    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        instance = response.data
        image_path = instance.get('image', None)
        
        if image_path:
            cnn_result = predict_cnn_result(image_path)
            instance['result'] = cnn_result

        return Response(instance)
