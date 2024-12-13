from openai import OpenAI
import os
from vlm3d import VLM3D


USE_GPT=False

if USE_GPT:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
else:
    api_key = "EMPTY"
    base_url = "http://localhost:8000/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
    
if __name__ == "__main__":
    point_file = '/Users/henryhe/code/3D-QA/pointvis/example/instrument.npy'
    vlm3d = VLM3D(client)
    prompt = "This is a point cloud of"
    print(vlm3d.response(prompt, point_file))
