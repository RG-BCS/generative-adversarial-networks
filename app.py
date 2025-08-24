# app.py (entry point for Render or HF Spaces)
import os
import uvicorn
from server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Render sets PORT automatically
    uvicorn.run(app, host="0.0.0.0", port=port)
