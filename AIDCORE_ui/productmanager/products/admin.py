from django.contrib import admin
from .models import Product, CustomUser

admin.site.register(Product)
admin.site.register(CustomUser)