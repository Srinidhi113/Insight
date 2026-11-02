"""Run from VS Code with F5 or from a terminal: python run.py."""
import os
import uvicorn
if __name__ == '__main__':
    uvicorn.run('insightflow.api:app', host='127.0.0.1', port=int(os.getenv('PORT', '8000')), workers=1)
