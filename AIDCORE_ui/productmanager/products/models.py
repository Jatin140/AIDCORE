from django.db import models
from django.contrib.auth.models import User

class Product(models.Model):
    name = models.CharField(max_length=200)
    review_image = models.ImageField(upload_to='reviews/')

    def __str__(self):
        return self.name

class CustomUser(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    telephone = models.CharField(max_length=15)
    products = models.ManyToManyField(Product, blank=True)

    def __str__(self):
        return self.user.username