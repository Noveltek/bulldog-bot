import os
import discord
from discord.ext import commands
import torch
import requests

# === CONFIG ===
TOKEN = os.getenv("DISCORD_TOKEN")
MODEL_URL = "https://github.com/Noveltek/bulldog-bot/releases/download/v1.0-model/roberta_bilstm_final.pt"
MODEL_PATH = "roberta_bilstm_final.pt"

# === DISCORD BOT SETUP ===
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# === LAZY MODEL LOADER ===
model = None

def load_model():
    global model
    if model is not None:
        return model

    # Download model file if missing
    if not os.path.exists(MODEL_PATH):
        print("🔄 Downloading model file from GitHub...")
        resp = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(resp.content)
        print("✅ Model downloaded.")

    print("⚡ Loading model gradually with memory optimizations...")

    # Import model class
    from model_def import RobertaBiLSTM
    model = RobertaBiLSTM()

    # Load weights gradually
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = model.state_dict()
    for name, param in checkpoint.items():
        if name in state_dict:
            state_dict[name].copy_(param)
    model.load_state_dict(state_dict)
    model.eval()

    # Convert to half precision
    for p in model.parameters():
        p.data = p.data.half()

    print("✅ Model fully loaded in half precision.")
    return model

# === COMMANDS ===
@bot.event
async def on_ready():
    print(f"Bulldog is online as {bot.user}")

@bot.command()
async def ping(ctx):
    await ctx.send("Pong 🐶")

@bot.command()
async def classify(ctx, *, text: str):
    mdl = load_model()

    # Example input (replace with tokenizer logic)
    input_ids = torch.randint(0, 100, (1, 10))  # dummy input
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = mdl(input_ids=input_ids, attention_mask=attention_mask)

    await ctx.send(f"Model output: {output}")
    
# === START BOT ===
bot.run(TOKEN)
