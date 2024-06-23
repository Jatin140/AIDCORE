from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.contrib.auth.models import User
from .models import Product, CustomUser
from .serializers import ProductSerializer, UserSerializer, CustomUserSerializer
import random


class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticated]

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            self.permission_classes = [IsAdminUser]
        return super().get_permissions()

    @action(detail=True, methods=['post'])
    def research(self, request, pk=None):
        product = self.get_object()
        features = ['Quality', 'Price', 'Durability', 'Design', 'Performance']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

        result = {
            'product_name': product.name,
            'features': features,
            'brands': brands,
            'ratings': [[random.randint(1, 5) for _ in range(5)] for _ in range(5)]
        }

        return Response(result)


class CustomUserViewSet(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer
    permission_classes = [IsAuthenticated]

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy', 'list']:
            self.permission_classes = [IsAdminUser]
        return super().get_permissions()

    @action(detail=False, methods=['get'])
    def me(self, request):
        serializer = self.get_serializer(request.user.customuser)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def add_product(self, request):
        user = request.user.customuser
        product_id = request.data.get('product_id')
        try:
            product = Product.objects.get(id=product_id)
            user.products.add(product)
            return Response({'status': 'product added'})
        except Product.DoesNotExist:
            return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)

    @action(detail=False, methods=['post'])
    def remove_product(self, request):
        user = request.user.customuser
        product_id = request.data.get('product_id')
        try:
            product = Product.objects.get(id=product_id)
            user.products.remove(product)
            return Response({'status': 'product removed'})
        except Product.DoesNotExist:
            return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)