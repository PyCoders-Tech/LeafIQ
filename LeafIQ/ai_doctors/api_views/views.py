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
        image_path = instance.get('image', None),
        tree_name = instance.get('name', None),
        test_type = instance.get('test_type', None)
        
        if image_path and tree_name and test_type:
            cnn_result = predict_cnn_result(tree_name, test_type, image_path)
            instance['result'] = cnn_result

        return Response(instance)
