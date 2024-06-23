from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Product, CustomUser

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'name', 'review_image']

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user

class CustomUserSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    products = ProductSerializer(many=True, read_only=True)

    class Meta:
        model = CustomUser
        fields = ['id', 'user', 'telephone', 'products']

    def create(self, validated_data):
        user_data = validated_data.pop('user')
        user = User.objects.create_user(**user_data)
        custom_user = CustomUser.objects.create(user=user, **validated_data)
        return custom_user