from dotenv import load_dotenv
load_dotenv()
import os
from mem0 import Memory
QUADRANT_HOST = "localhost"
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"
from openai import OpenAI

client = OpenAI(
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


config = {
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/embedding-001",
            "embedding_dims": 768,
            "api_key": GEMINI_API_KEY
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": 6333,
            "collection_name": "story",
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash",
            "temperature": 0.2,
            "max_tokens": 2000,
            "api_key": GEMINI_API_KEY
        }
    }
}

mem_client = Memory.from_config(config)


def chat(message):
    mem_result = mem_client.search(query=message, user_id="p123")
    # print(f"\n\n MEMORY: \n{mem_result}\n")
    
    memories = "\n".join(m["memory"] for m in mem_result.get("results"))
    # print(memories)

    SYSTEM_PROMPT = f"""
        You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
        systematically analyze input content, extract structured knowledge, and maintain an
        optimized memory store. Your primary function is information distillation
        and knowledge preservation with contextual awareness.

        Tone: Professional analytical, precision-focused, with clear uncertainty signaling
        
        Memory and Score:
        {memories}
    """


    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]
    result = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )

    messages.append(
        {"role": "assistant", "content": result.choices[0].message.content}
    )
    mem_client.add(messages, user_id="p123")
    return result.choices[0].message.content

while True:
    message = input(">> ")
    print("BOT: ", chat(message=message))