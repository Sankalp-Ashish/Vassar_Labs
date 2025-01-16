import os
import pprint
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

serper_api_key = os.environ.get("SERPER_API_KEY")

if not serper_api_key:
    raise ValueError("SERPER_API_KEY environment variable not set. Please set it in .env file.")

try:
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
    results = search.run("Obama's first name?")
    pprint.pprint(results)
except Exception as e:
    print(f"An error occurred: {e}")