import base64
import json
import yaml
from fastapi import FastAPI
from model.model import Llama3Model

# TODO: docstrings


with open("training-configs.yaml", encoding="utf8") as conf:
    args = yaml.load(conf, yaml.SafeLoader)

model = Llama3Model(args["model"], args["custom_checkpoint_path"], **args['training'])
app = FastAPI()


@app.get("/debug")
async def sanity_check():
    return {"message": "everything is fine"}


@app.post("/query")
async def chat(base64_converted_chat_history_json_and_settings: str):
    json_chat = base64.b64decode(base64_converted_chat_history_json_and_settings).decode()
    data = json.loads(json_chat)

    if not isinstance(data, dict):
        return json.dumps({"error": "invalid input format"})

    if "sentences" not in data.keys():
        return json.dumps({"error": "missing the conversation"})

    out_sentences = model.generate(**data)
    return {"status": "OK", "reply": out_sentences[0]}
