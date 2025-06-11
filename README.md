# FIR Legal Section Recommender

A full-stack application that uses AI to recommend relevant legal sections based on crime descriptions. The system consists of a React frontend and a Python Flask backend with machine learning capabilities.

## 🏗️ Architecture

- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: Python Flask API with NLTK and TF-IDF
- **ML Model**: TF-IDF Vectorization with Cosine Similarity

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Your FIR-DATA.csv file

### Backend Setup

1. **Install Python dependencies and setup NLTK data:**
   ```bash
   python3 setup_backend.py
   ```

2. **Start the Flask backend:**
   ```bash
   python3 app_simple.py
   ```
   
   The API will be available at `http://localhost:5001`

### Frontend Setup

1. **Navigate to the React project:**
   ```bash
   cd lawsmart-section-finder
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173`

## 📁 Project Structure

```
├── app_simple.py                     # Flask backend API (TF-IDF version)
├── app.py                           # Original Flask backend (sentence-transformers)
├── requirements.txt                 # Python dependencies
├── setup_backend.py                # Backend setup script
├── preprocess_data.pkl             # Preprocessed FIR data
├── fir_project(working).py         # Original Python script
├── lawsmart-section-finder/        # React frontend
│   ├── src/
│   │   ├── pages/Index.tsx         # Main application page
│   │   └── components/             # UI components
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🔧 API Endpoints

### POST /api/suggest
Analyzes a crime description and returns relevant legal sections.

**Request:**
```json
{
  "complaint": "My motorcycle was stolen from outside my office building..."
}
```

**Response:**
```json
[
  {
    "Description": "Whoever commits theft...",
    "Offense": "Theft",
    "Punishment": "Imprisonment of either description...",
    "Cognizable": "Yes",
    "Bailable": "No",
    "Court": "Magistrate Court"
  }
]
```

### GET /api/health
Health check endpoint to verify the backend is running.

## 🎯 Features

- **Real-time Analysis**: AI-powered legal section recommendations
- **Modern UI**: Clean, responsive interface with loading states
- **Error Handling**: Graceful error handling and user feedback
- **CORS Enabled**: Cross-origin requests supported
- **Health Monitoring**: Backend health check endpoint

## 🔄 Data Flow

1. User enters crime description in React frontend
2. Frontend sends POST request to Flask backend
3. Backend preprocesses text and runs TF-IDF similarity analysis
4. Backend returns relevant legal sections
5. Frontend displays results with proper formatting

## 🛠️ Development

### Backend Development
- Uses TF-IDF vectorization for text similarity
- Avoids dependency conflicts with sentence-transformers
- Model and data are loaded once at startup for performance
- CORS is enabled for frontend communication

### Frontend Development
- Updated to use real API instead of mock data
- Proper error handling for connection issues
- Loading states and user feedback

## 🐛 Troubleshooting

### Backend Issues
- Ensure FIR-DATA.csv is in the correct location
- Check that all Python dependencies are installed
- Verify NLTK data is downloaded
- Use `app_simple.py` to avoid sentence-transformers conflicts

### Frontend Issues
- Ensure backend is running on port 5001
- Check browser console for CORS errors
- Verify API endpoint URL is correct

### Connection Issues
- Backend must be running before starting frontend
- Check firewall settings for port 5001
- Ensure both servers are on localhost

## 📝 Notes

- The system uses TF-IDF vectorization for semantic similarity
- Preprocessed data is cached in `preprocess_data.pkl` for faster startup
- The similarity threshold is set to 0.1 (10%) for relevant matches
- Minimum 5 suggestions are returned per query
- Port 5001 is used to avoid conflicts with AirPlay on macOS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both frontend and backend
5. Submit a pull request 