#!/usr/bin/env python3
"""
Setup script for the FIR Legal Section Recommender backend
"""

import subprocess
import sys
import nltk

def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    try:
        # Install huggingface-hub first to resolve compatibility
        print("Installing huggingface-hub for compatibility...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub==0.16.4"])
        
        # Install core packages individually to avoid conflicts
        core_packages = [
            "flask==2.3.3",
            "flask-cors==4.0.0", 
            "sentence-transformers==2.2.2",
            "scikit-learn==1.3.0",
            "nltk==3.9.1"
        ]
        
        for package in core_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Package {package} may already be installed or have conflicts")
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("⚠️  Trying alternative installation method...")
        
        # Alternative: Install packages individually
        try:
            packages = [
                "huggingface-hub==0.16.4",
                "flask==2.3.3",
                "flask-cors==4.0.0",
                "sentence-transformers==2.2.2",
                "scikit-learn==1.3.0"
            ]
            
            for package in packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
                
        except subprocess.CalledProcessError as e2:
            print(f"❌ Alternative installation also failed: {e2}")
            return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def check_existing_packages():
    """Check if required packages are available"""
    print("Checking existing packages...")
    try:
        import flask
        import flask_cors
        import sentence_transformers
        import sklearn
        import pandas
        import torch
        import numpy
        import huggingface_hub
        print("✅ All required packages are available!")
        return True
    except ImportError as e:
        print(f"⚠️  Missing package: {e}")
        return False

def main():
    print("🚀 Setting up FIR Legal Section Recommender Backend")
    print("=" * 50)
    
    # Check if packages are already available
    if check_existing_packages():
        print("✅ Required packages are already installed!")
    else:
        # Install requirements
        if not install_requirements():
            print("❌ Setup failed during package installation")
            print("💡 You can try installing packages manually:")
            print("   pip install huggingface-hub==0.16.4 flask==2.3.3 flask-cors==4.0.0 sentence-transformers==2.2.2 scikit-learn==1.3.0")
            return
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Setup failed during NLTK data download")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Make sure your FIR-DATA.csv file is in the correct location")
    print("2. Run: python3 app.py")
    print("3. The API will be available at http://localhost:5000")
    print("4. Start your React frontend to connect to the backend")

if __name__ == "__main__":
    main() 