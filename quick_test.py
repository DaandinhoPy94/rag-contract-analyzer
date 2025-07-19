# quick_test.py
import sys
sys.path.append('.')

print("Testing basic imports...")

try:
    import google.generativeai as genai
    print("✅ Google Generative AI")
except Exception as e:
    print(f"❌ Google Generative AI: {e}")

try:
    from config import GeminiConfig
    print("✅ Config")
except Exception as e:
    print(f"❌ Config: {e}")

try:
    print("✅ Basic imports work")
    print("Now testing Gemini API...")
    
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        
        # Try different model names
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-pro'
        ]
        
        success = False
        for model_name in model_names:
            try:
                print(f"Trying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello")
                print(f"✅ Gemini API works with model: {model_name}")
                print(f"Response: {response.text[:50]}...")
                success = True
                break
            except Exception as e:
                print(f"❌ Model {model_name} failed: {e}")
                continue
        
        if not success:
            print("❌ All models failed, let's list available models:")
            try:
                models = genai.list_models()
                print("Available models:")
                for model in models:
                    print(f"  - {model.name}")
            except Exception as e:
                print(f"Could not list models: {e}")
        
    else:
        print("❌ No API key found")

except Exception as e:
    print(f"❌ Gemini test failed: {e}")