import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import {
  Button, Box, Typography, List, ListItem, ListItemText, TextField
} from '@mui/material';

function AdminDashboard() {
  console.log("Rendering AdminDashboard");
  const { logout, user } = useAuth();
  const [products, setProducts] = useState([]);
  const [newProduct, setNewProduct] = useState({ name: '', description: '' });

  useEffect(() => {
    console.log("AdminDashboard useEffect - Fetching products");
    fetchProducts();
  }, []);

  const fetchProducts = async () => {
    console.log("Fetching all products");
    try {
      const response = await axios.get('http://localhost:8000/api/products/', {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      console.log("Products fetched:", response.data);
      setProducts(response.data);
    } catch (error) {
      console.error('Error fetching products', error);
    }
  };

  const handleCreateProduct = async (e) => {
    e.preventDefault();
    console.log("Creating new product:", newProduct);
    try {
      const response = await axios.post('http://localhost:8000/api/products/', newProduct, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      console.log("Product created:", response.data);
      setNewProduct({ name: '', description: '' });
      fetchProducts();
    } catch (error) {
      console.error('Error creating product', error);
    }
  };

  console.log("AdminDashboard render - User:", user);
  console.log("AdminDashboard render - Products:", products);

  return (
    <Box sx={{ maxWidth: 800, margin: 'auto', mt: 5 }}>
      <Typography variant="h4" sx={{ mb: 2 }}>Admin Dashboard</Typography>
      <Button onClick={logout} variant="outlined" sx={{ mb: 2 }}>Logout</Button>

      <Typography variant="h5" sx={{ mb: 2 }}>Create New Product</Typography>
      <form onSubmit={handleCreateProduct}>
        <TextField
          fullWidth
          label="Product Name"
          value={newProduct.name}
          onChange={(e) => setNewProduct({ ...newProduct, name: e.target.value })}
          margin="normal"
          required
        />
        <TextField
          fullWidth
          label="Product Description"
          value={newProduct.description}
          onChange={(e) => setNewProduct({ ...newProduct, description: e.target.value })}
          margin="normal"
          required
        />
        <Button type="submit" variant="contained" sx={{ mt: 2 }}>Create Product</Button>
      </form>

      <Typography variant="h5" sx={{ mb: 2, mt: 4 }}>All Products</Typography>
      <List>
        {products.map((product) => (
          <ListItem key={product.id}>
            <ListItemText primary={product.name} secondary={product.description} />
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

export default AdminDashboard;