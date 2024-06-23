import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { List, ListItem, ListItemText, Button, TextField, Box } from '@mui/material';

function ProductList() {
  const [products, setProducts] = useState([]);
  const [newProduct, setNewProduct] = useState({ name: '', review_image: null });

  useEffect(() => {
    fetchProducts();
  }, []);

  const fetchProducts = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/products/', {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      setProducts(response.data);
    } catch (error) {
      console.error('Error fetching products', error);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('name', newProduct.name);
    formData.append('review_image', newProduct.review_image);

    try {
      await axios.post('http://localhost:8000/api/products/', formData, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'multipart/form-data'
        }
      });
      setNewProduct({ name: '', review_image: null });
      fetchProducts();
    } catch (error) {
      console.error('Error creating product', error);
    }
  };

  return (
    <Box>
      <form onSubmit={handleCreate}>
        <TextField
          label="Product Name"
          value={newProduct.name}
          onChange={(e) => setNewProduct({ ...newProduct, name: e.target.value })}
        />
        <input
          type="file"
          onChange={(e) => setNewProduct({ ...newProduct, review_image: e.target.files[0] })}
        />
        <Button type="submit">Create Product</Button>
      </form>
      <List>
        {products.map((product) => (
          <ListItem key={product.id}>
            <ListItemText primary={product.name} />
            <img src={product.review_image} alt={product.name} style={{width: 100, height: 100}} />
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

export default ProductList;