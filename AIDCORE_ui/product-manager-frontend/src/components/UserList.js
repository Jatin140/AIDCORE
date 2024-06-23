import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { List, ListItem, ListItemText, Button, TextField, Box } from '@mui/material';

function UserList() {
  const [users, setUsers] = useState([]);
  const [newUser, setNewUser] = useState({ username: '', email: '', password: '', telephone: '' });

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/users/', {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      setUsers(response.data);
    } catch (error) {
      console.error('Error fetching users', error);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    try {
      await axios.post('http://localhost:8000/api/users/', newUser, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      setNewUser({ username: '', email: '', password: '', telephone: '' });
      fetchUsers();
    } catch (error) {
      console.error('Error creating user', error);
    }
  };

  return (
    <Box>
      <form onSubmit={handleCreate}>
        <TextField
          label="Username"
          value={newUser.username}
          onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
        />
        <TextField
          label="Email"
          value={newUser.email}
          onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
        />
        <TextField
          label="Password"
          type="password"
          value={newUser.password}
          onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
        />
        <TextField
          label="Telephone"
          value={newUser.telephone}
          onChange={(e) => setNewUser({ ...newUser, telephone: e.target.value })}
        />
        <Button type="submit">Create User</Button>
      </form>
      <List>
        {users.map((user) => (
          <ListItem key={user.id}>
            <ListItemText primary={user.user.username} secondary={user.user.email} />
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

export default UserList;