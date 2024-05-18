from pydantic import BaseModel


class QueryPayload(BaseModel):
    """Used for the payload for /query"""

    base64_converted_chat_history_json_and_settings: str
