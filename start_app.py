"""
Easy launcher voor RAG Contract Analyzer
"""
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔧 Checking requirements...")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment")
        print("   Consider running: python -m venv venv && source venv/bin/activate")
    
    # Check if .env exists
    if not Path('.env').exists():
        print("❌ .env file not found")
        print("   Create .env with: GEMINI_API_KEY=your-api-key")
        return False
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError:
        print("❌ Streamlit not installed")
        print("   Run: pip install -r requirements.txt")
        return False
    
    # Check if our modules are available
    try:
        from config import GeminiConfig
        print("✅ Configuration module available")
    except ImportError:
        print("❌ Configuration module not found")
        return False
    
    print("✅ All requirements met!")
    return True

def main():
    """Main launcher function"""
    print("🚀 RAG Contract Analyzer Launcher")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    print("\n🎯 Starting Streamlit application...")
    print("📱 The app will open in your browser automatically")
    print("🔗 If not, go to: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    # Change to the correct directory
    os.chdir(Path(__file__).parent)
    
    # Start streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting application: {e}")
        print("\nTry running manually:")
        print("streamlit run frontend/app.py")

if __name__ == "__main__":
    main()
