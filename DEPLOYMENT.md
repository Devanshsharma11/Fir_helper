# 🚀 Deployment Guide - FIR Legal Section Recommender

This guide will help you deploy your FIR Legal Section Recommender application on Vercel and other platforms.

## 📋 **Deployment Strategy**

### **Recommended Approach:**
- **Frontend (React)**: Deploy on Vercel
- **Backend (Flask)**: Deploy on Railway/Render/Heroku
- **Database**: Use the preprocessed pickle file

## 🎯 **Step 1: Deploy Backend (Flask API)**

### **Option A: Railway (Recommended)**

1. **Create Railway Account:**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy Backend:**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login to Railway
   railway login
   
   # Initialize project
   railway init
   
   # Deploy
   railway up
   ```

3. **Set Environment Variables:**
   - Go to Railway Dashboard
   - Add environment variables if needed

### **Option B: Render**

1. **Create Render Account:**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Deploy Backend:**
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python3 app_deploy.py`
   - Set Python version: 3.12

### **Option C: Heroku**

1. **Create Heroku Account:**
   - Go to [heroku.com](https://heroku.com)
   - Sign up

2. **Deploy Backend:**
   ```bash
   # Install Heroku CLI
   # Create Heroku app
   heroku create your-app-name
   
   # Add buildpacks
   heroku buildpacks:set heroku/python
   
   # Deploy
   git push heroku main
   ```

## 🎯 **Step 2: Deploy Frontend (React) on Vercel**

### **Method 1: Vercel CLI**

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy Frontend:**
   ```bash
   cd lawsmart-section-finder
   vercel
   ```

3. **Set Environment Variables:**
   - Go to Vercel Dashboard
   - Add `VITE_API_URL` with your backend URL

### **Method 2: Vercel Dashboard**

1. **Connect GitHub:**
   - Go to [vercel.com](https://vercel.com)
   - Connect your GitHub repository

2. **Configure Project:**
   - Set root directory: `lawsmart-section-finder`
   - Set build command: `npm run build`
   - Set output directory: `dist`

3. **Set Environment Variables:**
   - `VITE_API_URL`: Your backend URL (e.g., `https://your-app.railway.app`)

## 🔧 **Configuration Files**

### **Backend Files:**
- `app_deploy.py` - Production Flask app
- `requirements.txt` - Python dependencies
- `Procfile` - For Heroku/Railway
- `runtime.txt` - Python version
- `preprocess_data.pkl` - Your FIR data

### **Frontend Files:**
- `lawsmart-section-finder/vercel.json` - Vercel configuration
- `lawsmart-section-finder/package.json` - React dependencies

## 🌐 **Environment Variables**

### **Frontend (Vercel):**
```env
VITE_API_URL=https://your-backend-url.railway.app
```

### **Backend (Railway/Render/Heroku):**
```env
PORT=5001
```

## 📁 **File Structure for Deployment**

```
├── app_deploy.py                    # Production Flask app
├── requirements.txt                 # Python dependencies
├── Procfile                        # For Heroku/Railway
├── runtime.txt                     # Python version
├── preprocess_data.pkl             # FIR data
├── lawsmart-section-finder/        # React frontend
│   ├── vercel.json                # Vercel config
│   ├── package.json               # React dependencies
│   └── src/
└── DEPLOYMENT.md                   # This guide
```

## 🚀 **Quick Deploy Commands**

### **Backend (Railway):**
```bash
railway login
railway init
railway up
```

### **Frontend (Vercel):**
```bash
cd lawsmart-section-finder
vercel
```

## 🔍 **Testing Deployment**

### **Backend Health Check:**
```bash
curl https://your-backend-url.railway.app/api/health
```

### **Frontend Test:**
- Visit your Vercel URL
- Enter a crime description
- Check if it connects to your backend

## 🐛 **Troubleshooting**

### **Backend Issues:**
- Check Railway/Render logs
- Ensure `preprocess_data.pkl` is included
- Verify Python version compatibility

### **Frontend Issues:**
- Check Vercel build logs
- Verify `VITE_API_URL` environment variable
- Test API connectivity

### **CORS Issues:**
- Backend has CORS enabled
- Check if frontend URL is allowed

## 📝 **Important Notes**

1. **Data File**: Ensure `preprocess_data.pkl` is in your repository
2. **Environment Variables**: Set `VITE_API_URL` in Vercel
3. **CORS**: Backend allows all origins for development
4. **Port**: Backend uses `PORT` environment variable
5. **Build**: Frontend builds to `dist` directory

## 🎉 **Success Indicators**

- ✅ Backend responds to health check
- ✅ Frontend loads without errors
- ✅ API calls work from frontend
- ✅ Legal suggestions are returned

Your FIR Legal Section Recommender will be live and accessible worldwide! 🌍 