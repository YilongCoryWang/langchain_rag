from dotenv import load_dotenv
import os

load_dotenv()  # read .env file

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")