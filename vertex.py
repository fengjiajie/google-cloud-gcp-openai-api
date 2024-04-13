#!/usr/bin/env python3

# Copyright 2023 Nils Knieling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import secrets
import time
import datetime
import uvicorn

# FastAPI
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Google Vertex AI
import google.auth
from google.cloud import aiplatform

# LangChain
import langchain
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Google authentication
credentials, project_id = google.auth.default()

# Get environment variable
host = os.environ.get("HOST", "0.0.0.0")
port = int(os.environ.get("PORT", 8000))
debug = os.environ.get("DEBUG", False)
print(f"Endpoint: http://{host}:{port}/")
# Google Cloud
project = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", project_id)
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
print(f"Google Cloud project identifier: {project}")
print(f"Google Cloud location: {location}")
# LLM chat model name to use
model_name = os.environ.get("MODEL_NAME", "gemini-pro")
print(f"LLM chat model name: {model_name}")
# Token limit determines the maximum amount of text output from one prompt
default_max_output_tokens = os.environ.get("MAX_OUTPUT_TOKENS", "8192")
# Sampling temperature,
# it controls the degree of randomness in token selection
default_temperature = os.environ.get("TEMPERATURE", "0.7")
# How the model selects tokens for output, the next token is selected from
default_top_k = os.environ.get("TOP_K", "1")
# Tokens are selected from most probable to least until the sum of their
default_top_p = os.environ.get("TOP_P", "1")
# API key
default_api_key = f"sk-{secrets.token_hex(21)}"
api_key = os.environ.get("OPENAI_API_KEY", default_api_key)
print(f"API key: {api_key}")

app = FastAPI(
    title='OpenAI API',
    description='APIs for sampling from and fine-tuning language models',
    version='2.0.0',
    servers=[{'url': 'https://api.openai.com/'}],
    contact={
        "name": "GitHub",
        "url": "https://github.com/Cyclenerd/google-cloud-gcp-openai-api",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

aiplatform.init(
    project=project,
    location=location,
)


class Message(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    messages: List[Message]
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]


@app.get("/")
def read_root():
    return {
        "LangChain": langchain.__version__,
        "Vertex AI": aiplatform.__version__
    }


@app.get("/v1/models")
def get_models():
    """
    Lists the currently available models,
    and provides basic information about each one
    such as the owner and availability.

    https://platform.openai.com/docs/api-reference/models/list
    """
    id = f"modelperm-{secrets.token_hex(12)}"
    ts = int(time.time())
    models = {"data": [], "object": "list"}
    models['data'].append({
        "id": "gpt-3.5-turbo",
        "object": "model",
        "created": ts,
        "owned_by": "openai",
        "permission": [
            {
                "id": id,
                "created": ts,
                "object": "model_permission",
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False
            }
        ],
        "root": "gpt-3.5-turbo",
        "parent": None,
    })
    models['data'].append({
        "id": "text-embedding-ada-002",
        "object": "model",
        "created": ts,
        "owned_by": "openai-internal",
        "permission": [
            {
                "id": id,
                "created": ts,
                "object": "model_permission",
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": True,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False
            }
        ],
        "root": "text-embedding-ada-002",
        "parent": None
    })
    return models


def generate_stream_response_start():
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "created": ts,
        "object": "chat.completion.chunk",
        "model": "gpt-3.5-turbo",
        "choices": [{
            "delta": {"role": "assistant"},
            "index": 0,
            "finish_reason": None
        }]
    }


def generate_stream_response(content: str):
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "created": ts,
        "object": "chat.completion.chunk",
        "model": "gpt-3.5-turbo",
        "choices": [{
            "delta": {"content": content},
            "index": 0,
            "finish_reason": None
        }]
    }


def generate_stream_response_stop():
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "created": ts,
        "object": "chat.completion.chunk",
        "model": "gpt-3.5-turbo",
        "choices": [{
            "delta": {},
            "index": 0,
            "finish_reason": "stop"
        }]
    }


def generate_response(content: str):
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "created": ts,
        "object": "chat.completion",
        "model": "gpt-3.5-turbo",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop", "index": 0}
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatBody, request: Request):
    """
    Creates a model response for the given chat conversation.

    https://platform.openai.com/docs/api-reference/chat/create
    """

    # Authorization via OPENAI_API_KEY
    if request.headers.get("Authorization").split(" ")[1] != api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key is wrong!")

    if debug:
        print(f"body = {body}")

    # Get user question
    question = body.messages[-1]
    if question.role == 'user' or question.role == 'assistant':
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    # Overwrite defaults
    temperature = float(body.temperature or default_temperature)
    top_k = int(default_top_k)
    top_p = float(body.top_p or default_top_p)
    max_output_tokens = int(body.max_tokens or default_max_output_tokens)
    # Note: Max output token:
    # - gemini-pro: 8192
    #   https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
    # - chat-bison: 1024
    # - codechat-bison: 2048
    # - ..-32k: The total amount of input and output tokens adds up to 32k.
    #           For example, if you specify 16k of input tokens,
    #           then you can receive up to 16k of output tokens.
    if model_name == 'codechat-bison':
        if max_output_tokens > 2048:
            max_output_tokens = 2048
    elif model_name.find("gemini-pro"):
        if max_output_tokens > 8192:
            max_output_tokens = 8192
    elif model_name.find("32k"):
        if max_output_tokens > 16000:
            max_output_tokens = 16000
    elif max_output_tokens > 1024:
        max_output_tokens = 1024

    # Wrapper around Vertex AI large language models
    llm = ChatVertexAI(
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_output_tokens=max_output_tokens
    )

    # Buffer for storing conversation memory
    # Note: Max input token:
    # - chat-bison: 4096
    # - codechat-bison: 6144
    memory = ConversationBufferMemory(
        memory_key="history",
        max_token_limit=30720,
        return_messages=True
    )
    # Today
    #memory.chat_memory.add_user_message("What day is today?")
    #memory.chat_memory.add_ai_message(
    #    datetime.date.today().strftime("Today is %A, %B %d, %Y")
    #)

    memory.chat_memory.add_user_message("你是 Gemini，一个由谷歌训练的基于大型语言模型的聊天机器人。你的主要任务是对用户的提问给出详实且精确的回答。请仔细分析用户的话语，对于复杂问题可以将逐步思考的过程输出给用户。使用中文与用户对话。")
    memory.chat_memory.add_ai_message("你好！我是 Gemini，很高兴认识你。我会尽力理解你的问题，并给出详实而精确的答案。我会仔细分析你的话语，并尝试理解你的意图。对于复杂的问题，我会分享我的思考过程，以便你更好地理解我的答案。")
    # Add history
    for message in body.messages:
        #if message.role == 'system':
        #    memory.chat_memory.add_user_message(message.content)
        #    memory.chat_memory.add_ai_message("好的。")
        if message.role == 'user':
            memory.chat_memory.add_user_message(message.content)
        elif message.role == 'assistant':
            memory.chat_memory.add_ai_message(message.content)

    # Get Vertex AI output
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
    )
    answer = conversation.predict(input=question)

    if debug:
        print(f"stream = {body.stream}")
        print(f"model = {body.model}")
        print(f"temperature = {temperature}")
        print(f"top_k = {top_k}")
        print(f"top_p = {top_p}")
        print(f"max_output_tokens = {max_output_tokens}")
        print(f"history = {memory.buffer}")

    # Return output
    if body.stream:
        async def stream():
            yield json.dumps(
                generate_stream_response_start(),
                ensure_ascii=False
            )
            yield json.dumps(
                generate_stream_response(answer),
                ensure_ascii=False
            )
            yield json.dumps(
                generate_stream_response_stop(),
                ensure_ascii=False
            )
        return EventSourceResponse(stream(), ping=10000)
    else:
        return JSONResponse(content=generate_response(answer))

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
