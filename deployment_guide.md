# Deployment Guide for AI Agent Simulation Dashboard

## Quick Deploy Options

### 1. Streamlit Cloud (Recommended - Free)
**Best for:** Academic projects, quick deployment, no server management

```bash
# 1. Push your code to GitHub
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main

# 2. Deploy on Streamlit Cloud
# - Go to https://share.streamlit.io/
# - Sign in with GitHub
# - Click "New app"
# - Select your repository and set app.py as the main file
# - Deploy!
```

**Requirements file needed:**
```bash
# Create requirements.txt
pip freeze > requirements.txt
```

### 2. Heroku (Free tier discontinued, but still popular)
**Best for:** Full control, custom domains

```bash
# 1. Install Heroku CLI
# 2. Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# 3. Create runtime.txt
echo "python-3.11.0" > runtime.txt

# 4. Deploy
heroku create your-app-name
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### 3. Railway (Free tier available)
**Best for:** Easy deployment, good free tier

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway init
railway up
```

### 4. Google Cloud Run (Free tier)
**Best for:** Scalable, pay-per-use

```bash
# 1. Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
EOF

# 2. Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/streamlit-app
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/streamlit-app --platform managed
```

### 5. Local Network Sharing
**Best for:** Quick sharing with colleagues

```bash
# Run on your local machine, accessible to others on same network
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## Required Files for Deployment

### 1. requirements.txt
```bash
# Generate from your current environment
pip freeze > requirements.txt
```

### 2. .streamlit/config.toml (Optional)
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### 3. .gitignore (Update)
```gitignore
# Add these to your existing .gitignore
__pycache__/
*.pyc
.env
.venv/
venv/
prosocial_analysis_env/
```

## Pre-Deployment Checklist

### ✅ Code Preparation
- [ ] All file paths use relative paths (✅ Already done)
- [ ] No hardcoded local file paths
- [ ] Environment variables for sensitive data (if any)
- [ ] Requirements.txt is up to date

### ✅ Data Files
- [ ] Ensure data files are in the repository
- [ ] Check file sizes (large files may need Git LFS)
- [ ] Verify all required config files exist

### ✅ Testing
- [ ] Test locally with `streamlit run app.py`
- [ ] Verify all features work
- [ ] Check for any missing dependencies

## Recommended Approach for Academic Use

**For your academic project, I recommend Streamlit Cloud because:**

1. **Free and Easy**: No credit card required, simple GitHub integration
2. **Academic-Friendly**: Perfect for research projects and papers
3. **Automatic Updates**: Deploys automatically when you push to GitHub
4. **No Server Management**: Handles all infrastructure for you
5. **Public/Private**: Can be public or private depending on your needs

## Step-by-Step Streamlit Cloud Deployment

```bash
# 1. Create GitHub repository
# Go to github.com and create a new repository

# 2. Initialize git and push code
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main

# 3. Generate requirements.txt
pip freeze > requirements.txt

# 4. Deploy on Streamlit Cloud
# - Visit https://share.streamlit.io/
# - Sign in with GitHub
# - Click "New app"
# - Repository: YOUR_USERNAME/YOUR_REPO_NAME
# - Main file path: app.py
# - Click "Deploy!"
```

## Troubleshooting Common Issues

### Port Issues
```bash
# If you get port errors, use:
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Missing Dependencies
```bash
# Update requirements.txt
pip install streamlit pandas numpy scipy plotly seaborn matplotlib pyarrow openpyxl joblib pyyaml
pip freeze > requirements.txt
```

### File Path Issues
```bash
# Ensure all paths are relative to project root
# Your current setup already does this correctly
```

## Security Considerations

1. **Data Privacy**: If using sensitive data, consider private deployment
2. **Environment Variables**: Use `.env` files for any API keys or secrets
3. **Access Control**: Streamlit Cloud allows you to control who can access your app

## Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| Streamlit Cloud | ✅ Unlimited | $10/month | Academic projects |
| Heroku | ❌ Discontinued | $7/month | Full control |
| Railway | ✅ $5 credit | Pay-per-use | Easy deployment |
| Google Cloud Run | ✅ 2M requests | Pay-per-use | Scalable |
| Local Network | ✅ Free | N/A | Quick sharing |

## Next Steps

1. **Choose your platform** (I recommend Streamlit Cloud)
2. **Prepare your repository** (add requirements.txt)
3. **Deploy and test**
4. **Share the URL** with your professor/colleagues

Would you like me to help you with any specific deployment platform? 