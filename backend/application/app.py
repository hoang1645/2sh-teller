import base64
import json
import yaml
import logging
import time
from uuid import uuid4
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pydantic_models import *

from model.model import Llama3Model


# log
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(f'app_{uuid4()}.log'), logging.StreamHandler()],
    )


configure_logging()


# Define logging middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request details
        logging.info(f"Request: {request.method} {request.url}")

        # Process the request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response details
        logging.info(f"Response: {response.status_code}")
        logging.info(f"Process time: {process_time:.4f} sec")

        return response


# initialize
with open("../infer-configs.yaml", encoding="utf8") as conf:
    args = yaml.load(conf, yaml.SafeLoader)

model = Llama3Model(args["model"], args["custom_checkpoint_path"], **args['training'])
app = FastAPI()
app.add_middleware(LoggingMiddleware)


@app.get("/debug")
async def sanity_check():
    return {"message": "everything is fine"}


@app.post("/query")
async def chat(payload: QueryPayload):
    try:
        json_chat = base64.b64decode(
            payload.base64_converted_chat_history_json_and_settings
        ).decode()
        logging.info(json_chat)
        data = json.loads(json_chat)
    except Exception as e:
        logging.error("bad payload: " + str(e))
        return {"status": "4xx", "error": "bad payload: " + str(e)}

    if not isinstance(data, dict):
        logging.error("invalid input format")
        return json.dumps({"error": "invalid input format"})

    if "sentences" not in data.keys():
        logging.error("missing the conversation")
        return json.dumps({"error": "missing the conversation"})

    try:
        out_sentences = model.generate(**data)
    except Exception as e:
        logging.critical(str(e))
        return {"status": "50x", "error": str(e)}
    logging.info("done")
    return {"status": "OK", "reply": out_sentences[0]}
    # debug
    return {'status': "OK", **data}
