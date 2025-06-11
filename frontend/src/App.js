import React, { useState } from 'react';
import './App.css';

function App() {
  const [complaint, setComplaint] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Use deployed Render backend
  const API_BASE_URL = 'https://fir-legal-backend.onrender.com';

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!complaint.trim()) {
      setError('Please enter a complaint');
      return;
    }

    setLoading(true);
    setError('');
    setSuggestions([]);

    try {
      console.log('Sending request to:', `${API_BASE_URL}/api/suggest`);
      console.log('Request data:', { complaint: complaint });
      
      const response = await fetch(`${API_BASE_URL}/api/suggest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ complaint: complaint }),
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      setSuggestions(data);
    } catch (err) {
      console.error('Fetch error:', err);
      setError('Error fetching suggestions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Test connection on component mount
  React.useEffect(() => {
    const testConnection = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
          const data = await response.json();
          console.log('Backend health check successful:', data);
        } else {
          console.error('Backend health check failed:', response.status);
        }
      } catch (err) {
        console.error('Backend connection test failed:', err);
      }
    };
    
    testConnection();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>FIR Legal Section Recommender</h1>
        <p>Enter your complaint to get relevant legal sections and punishments</p>
        <p style={{fontSize: '0.9rem', opacity: 0.8}}>Using deployed backend (Render)</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="complaint-form">
          <div className="form-group">
            <label htmlFor="complaint">Describe your complaint:</label>
            <textarea
              id="complaint"
              value={complaint}
              onChange={(e) => setComplaint(e.target.value)}
              placeholder="Enter your complaint details here..."
              rows="4"
              required
            />
          </div>
          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing...' : 'Get Legal Suggestions'}
          </button>
        </form>

        {error && (
          <div className="error">
            <p>{error}</p>
          </div>
        )}

        {suggestions.length > 0 && (
          <div className="suggestions">
            <h2>Legal Section Suggestions</h2>
            {suggestions.map((suggestion, index) => (
              <div key={index} className="suggestion-card">
                <h3>Offense: {suggestion.Offense}</h3>
                <div className="suggestion-details">
                  <p><strong>Description:</strong> {suggestion.Description}</p>
                  <p><strong>Punishment:</strong> {suggestion.Punishment}</p>
                  <p><strong>Cognizable:</strong> {suggestion.Cognizable}</p>
                  <p><strong>Bailable:</strong> {suggestion.Bailable}</p>
                  <p><strong>Court:</strong> {suggestion.Court}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
