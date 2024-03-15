from django.db import models

class TreeType(models.Model):
    TREE_NAME = {
        'potato': 'Potato',
    }
    TEST_TYPE = {
        'leaf': 'Leaf'
    }
    name = models.CharField(max_length=40, choices=TREE_NAME, default='draft',)
    
    test_type = models.CharField(max_length=40, choices=TEST_TYPE, default='draft',) # skin or leaf
    image = models.ImageField(upload_to='img/')

    def __str__(self):
        return f'{self.pk}'


