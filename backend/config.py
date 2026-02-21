import os
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(ROOT_DIR, '..', '.env'))

DATA_DIR = os.path.join(ROOT_DIR, '..', 'data', 'sampledocs')
PERSIST_DIR = os.path.join(ROOT_DIR, '..', 'vectordb')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")