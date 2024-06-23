import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useApiKey } from '../context/ApiKeyContext';
import axios from 'axios';
import { OpenAI } from 'openai';
import {
  Button, Box, Typography, List, ListItem, ListItemText, Dialog,
  DialogTitle, DialogContent, DialogActions, Grid, Rating, Checkbox, FormGroup, FormControlLabel, TextField
} from '@mui/material';
import { RadialBarChart, RadialBar, Legend, ResponsiveContainer } from 'recharts';

function UserDashboard() {
  const { logout } = useAuth();
  const { apiKey, setApiKey } = useApiKey();
  const [products, setProducts] = useState([]);
  const [allProducts, setAllProducts] = useState([]);
  const [open, setOpen] = useState(false);
  const [researchData, setResearchData] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState({});
  const [generatedAd, setGeneratedAd] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const [editingProduct, setEditingProduct] = useState(null);

  useEffect(() => {
    fetchUserProducts();
    fetchAllProducts();
  }, []);

  const fetchUserProducts = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/users/me/', {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      setProducts(response.data.products);
    } catch (error) {
      console.error('Error fetching user products', error);
    }
  };

  const fetchAllProducts = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/products/', {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      setAllProducts(response.data);
    } catch (error) {
      console.error('Error fetching all products', error);
    }
  };

  const handleResearch = async (productId) => {
    try {
      const response = await axios.post(`http://localhost:8000/api/products/${productId}/research/`, {}, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });

      const formattedData = response.data.features.map((feature, index) => ({
        name: feature,
        value: response.data.ratings[0][index],
        fill: `hsl(${index * 60}, 70%, 50%)`
      }));

      setResearchData({
        productName: response.data.product_name,
        overallRating: (response.data.ratings.flat().reduce((a, b) => a + b, 0) / response.data.ratings.flat().length).toFixed(1),
        chartData: formattedData
      });

      const initialSelectedFeatures = {};
      response.data.features.forEach(feature => {
        initialSelectedFeatures[feature] = false;
      });
      setSelectedFeatures(initialSelectedFeatures);

      setOpen(true);
    } catch (error) {
      console.error('Error researching product', error);
    }
  };

  const handleAddProduct = async (productId) => {
    try {
      await axios.post('http://localhost:8000/api/users/add_product/',
        { product_id: productId },
        { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` } }
      );
      fetchUserProducts();
    } catch (error) {
      console.error('Error adding product to user', error);
    }
  };

  const handleEditProduct = (product) => {
    setEditingProduct(product);
  };

const handleSaveEdit = async () => {
  try {
    const response = await axios.put(
      `http://localhost:8000/api/products/${editingProduct.id}/`,
      { name: editingProduct.name },
      { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` } }
    );

    if (response.status === 200) {
      // Update the local state with the edited product
      setProducts(prevProducts =>
        prevProducts.map(p => p.id === editingProduct.id ? {...p, name: editingProduct.name} : p)
      );
      setAllProducts(prevProducts =>
        prevProducts.map(p => p.id === editingProduct.id ? {...p, name: editingProduct.name} : p)
      );
      setEditingProduct(null);
    } else {
      console.error('Failed to update product:', response);
      alert('Failed to update product. Please try again.');
    }
  } catch (error) {
    console.error('Error saving edited product', error);
    alert('Error saving product. Please try again.');
  }
};

  const handleDeleteProduct = async (productId) => {
    try {
      await axios.post('http://localhost:8000/api/users/remove_product/',
        { product_id: productId },
        { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` } }
      );
      fetchUserProducts();
    } catch (error) {
      console.error('Error deleting product from user', error);
    }
  };

  const generateAdvertisement = async () => {
    let currentApiKey = apiKey || tempApiKey;
    if (!currentApiKey) {
      alert('Please enter your OpenAI API key to generate an advertisement.');
      return;
    }

    const openai = new OpenAI({ apiKey: currentApiKey, dangerouslyAllowBrowser: true });

    const selectedFeaturesList = Object.entries(selectedFeatures)
      .filter(([_, isSelected]) => isSelected)
      .map(([feature, _]) => feature);

    if (selectedFeaturesList.length === 0) {
      alert('Please select at least one feature.');
      return;
    }

    const prompt = `Create a short advertisement for ${researchData.productName} highlighting the following features: ${selectedFeaturesList.join(', ')}. Use the ratings provided to emphasize strengths: ${JSON.stringify(researchData.chartData)}`;

    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
      });

      setGeneratedAd(completion.choices[0].message.content);
      if (tempApiKey && !apiKey) {
        setApiKey(tempApiKey);
      }
    } catch (error) {
      console.error('Error generating advertisement:', error);
      setGeneratedAd('Failed to generate advertisement. Please check your OpenAI API key and try again.');
    }
  };

  return (
    <Box sx={{ maxWidth: 800, margin: 'auto', mt: 5 }}>
      <Typography variant="h4" sx={{ mb: 2 }}>User Dashboard</Typography>
      <Button onClick={logout} variant="outlined" sx={{ mb: 2 }}>Logout</Button>

      <Typography variant="h5" sx={{ mb: 2 }}>Your Products</Typography>
      <List>
        {products.map((product) => (
          <ListItem key={product.id}>
            <ListItemText primary={product.name} />
            <Button onClick={() => handleResearch(product.id)}>Research</Button>
            <Button onClick={() => handleEditProduct(product)}>Edit</Button>
            <Button onClick={() => handleDeleteProduct(product.id)}>Delete</Button>
          </ListItem>
        ))}
      </List>

      <Typography variant="h5" sx={{ mb: 2, mt: 4 }}>All Available Products</Typography>
      <List>
        {allProducts.map((product) => (
          <ListItem key={product.id}>
            <ListItemText primary={product.name} />
            <Button onClick={() => handleAddProduct(product.id)}>Add to My Products</Button>
          </ListItem>
        ))}
      </List>

      <Dialog open={!!editingProduct} onClose={() => setEditingProduct(null)}>
        <DialogTitle>Edit Product</DialogTitle>
        <DialogContent>
          {editingProduct && (
            <TextField
              fullWidth
              label="Product Name"
              value={editingProduct.name}
              onChange={(e) => setEditingProduct({...editingProduct, name: e.target.value})}
              margin="normal"
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditingProduct(null)}>Cancel</Button>
          <Button onClick={handleSaveEdit}>Save</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Research Results for {researchData?.productName}</DialogTitle>
        <DialogContent>
          {researchData && (
            <Box>
              <Grid container spacing={2} alignItems="center">
                <Grid item>
                  <Typography variant="h3">{researchData.overallRating}</Typography>
                </Grid>
                <Grid item>
                  <Rating value={parseFloat(researchData.overallRating)} readOnly precision={0.1} max={5} />
                </Grid>
              </Grid>
              <Box sx={{ height: 300, mt: 2 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="10%"
                    outerRadius="80%"
                    data={researchData.chartData}
                    startAngle={180}
                    endAngle={0}
                  >
                    <RadialBar minAngle={15} label={{ fill: '#666', position: 'insideStart' }} background clockWise={true} dataKey='value' />
                    <Legend iconSize={10} width={120} height={140} layout='vertical' verticalAlign='middle' align="right" />
                  </RadialBarChart>
                </ResponsiveContainer>
              </Box>
              <Typography variant="h6" sx={{ mt: 2 }}>Select features for advertisement:</Typography>
              <FormGroup>
                {researchData.chartData.map((item) => (
                  <FormControlLabel
                    key={item.name}
                    control={<Checkbox checked={selectedFeatures[item.name]} onChange={() => setSelectedFeatures({...selectedFeatures, [item.name]: !selectedFeatures[item.name]})} />}
                    label={item.name}
                  />
                ))}
              </FormGroup>
              {!apiKey && (
                <TextField
                  fullWidth
                  label="OpenAI API Key"
                  value={tempApiKey}
                  onChange={(e) => setTempApiKey(e.target.value)}
                  margin="normal"
                />
              )}
              <Button onClick={generateAdvertisement} variant="contained" sx={{ mt: 2 }}>Generate Advertisement</Button>
              {generatedAd && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="h6">Generated Advertisement:</Typography>
                  <Typography>{generatedAd}</Typography>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default UserDashboard;