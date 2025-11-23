import discord
import torch
import os
import requests

# --- Model setup ---
MODEL_URL = "https://github.com/Noveltek/bulldog-bot/releases/download/v1.0-model/roberta_bilstm_final.pt"
MODEL_PATH = "roberta_bilstm_final.pt"

# ✅ Updated: download from GitHub if file is missing
if not os.path.exists(MODEL_PATH):
    print("🔄 Downloading model file from GitHub...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded.")

# ✅ Load the model
try:
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# --- Discord bot setup ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    print("Ready!")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower() == "ping":
        await message.channel.send("Pong 🐶")

    # Add your model-based response logic here
    # Example: use model to generate reply based on message.content

# --- Run the bot ---
TOKEN = os.getenv("DISCORD_TOKEN")
client.run(TOKEN)
