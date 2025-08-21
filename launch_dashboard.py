#!/usr/bin/env python3
"""
Quick launcher for the AI Agent Simulation Dashboard
"""
import subprocess
import sys
from pathlib import Path

def main():
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    
    print("🚀 Starting AI Agent Simulation Dashboard...")
    print(f"📁 Project directory: {project_root}")
    print("🌐 Dashboard will open in your browser automatically")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())