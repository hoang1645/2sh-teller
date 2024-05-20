import json
import base64

import requests
import streamlit as st


beam_size = st.slider("Beam size", min_value=1, max_value=10, value=4)
top_p = st.slider(
    "Nucleus sampling $p$",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
)
top_k = st.number_input(
    "Top-k sampling $k$",
    min_value=5,
    max_value=500,
    value=50,
)
temperature = st.number_input(
    "Temperature",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
)


if prompt := st.chat_input("Your prompt here..."):
    st.chat_message("user").write(prompt)
    messages = [
        {
            "role": "system",
            "content": """You are a two-sentence horror storyteller.
        Given the user's one sentence, continue with exactly one sentence to make a two-sentence horror story.
        When the user inputs more than one sentence, prompt the user to use one sentence only and do not do anything else.""",
        },
        {"role": "user", "content": prompt},
    ]
    payload = {
        "sentences": messages,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "beam_size": beam_size,
    }

    payload_b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()
    rq = requests.post(
        "http://localhost:8000/query",
        json={"base64_converted_chat_history_json_and_settings": payload_b64},
        headers={"Content-Type": "application/json"},
    )
    response = json.loads(rq.text)
    print(response)
    reply = json.loads(response['reply'])  # remove the "assistant" part before the json
    # reply = json.loads(reply)
    st.chat_message("assistant").write(reply['content'])
