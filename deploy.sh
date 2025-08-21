#!/bin/bash

# Deployment script for AI Agent Simulation Dashboard
echo "🚀 Setting up deployment for AI Agent Simulation Dashboard"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: AI Agent Simulation Dashboard"
    git branch -M main
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "📦 Creating requirements.txt..."
    source prosocial_analysis_env/bin/activate && pip freeze > requirements.txt
    echo "✅ Requirements.txt created"
else
    echo "✅ Requirements.txt already exists"
fi

# Check if .gitignore exists and update it
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
prosocial_analysis_env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Outputs (keep this as it's already in your .gitignore)
outputs/
EOF
    echo "✅ .gitignore created"
else
    echo "✅ .gitignore already exists"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Create a GitHub repository at https://github.com/new"
echo "2. Run these commands (replace YOUR_USERNAME and REPO_NAME):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
echo "   git push -u origin main"
echo ""
echo "3. Deploy on Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io/"
echo "   - Sign in with GitHub"
echo "   - Click 'New app'"
echo "   - Select your repository"
echo "   - Set main file path to: app.py"
echo "   - Click 'Deploy!'"
echo ""
echo "📚 See deployment_guide.md for detailed instructions"
echo "✅ Setup complete!" 