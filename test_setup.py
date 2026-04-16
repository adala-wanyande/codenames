import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

print("✅ Numpy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)

# Check if the API keys are loading properly
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    print("✅ Gemini API Key found!")
else:
    print("❌ Gemini API Key missing!")