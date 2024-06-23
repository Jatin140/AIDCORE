import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function PrivateRoute({ children }) {
  const { user } = useAuth();

  console.log("PrivateRoute - User:", user); // Add this line

  if (!user) {
    console.log("PrivateRoute - Redirecting to login"); // Add this line
    return <Navigate to="/login" replace />;
  }

  console.log("PrivateRoute - Rendering children"); // Add this line
  return children;
}

export default PrivateRoute;