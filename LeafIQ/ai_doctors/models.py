from django.db import models

class TreeType(models.Model):
    name = models.CharField(max_length=40)
    test_type = models.CharField(max_length=40) # skin or leaf
    image = models.ImageField(upload_to='img/')

    def __str__(self):
        return f'{self.pk}'


