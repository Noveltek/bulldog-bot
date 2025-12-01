import os
import discord
from discord.ext import commands
from discord import app_commands
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from typing import Tuple

# ==============================================================
# 🛠️ 1. CONFIGURATION: WHAT TO CHANGE
# ==============================================================

# 🚨 IMPORTANT: 1. Replace this placeholder token.
DISCORD_TOKEN = "YOUR_BOT_DISCORD_TOKEN_HERE" 

# ➡️ 2. TOKENIZER PATH IS NOW SET TO THE ROOT DIRECTORY ('.')
# This matches your current file structure where the files are next to bulldog.py
TOKENIZER_PATH = "." 

# 🚨 3. Your Model File Details (The download link you provided) 🚨
MODEL_FILE_ID = "1BqLo-S8pXkaP8Mf7-JjPiNLI5odeelG1"
MODEL_PATH = "roberta_bilstm_final.pt"

# Model Constants
MAX_SEQ_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable for the moderator channel ID
MOD_ALERT_CHANNEL_ID: int = None

# ==============================================================
# 🧠 2. MODEL ARCHITECTURE & PREDICTION LOGIC
# ==============================================================

class RoBERTa_BiLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        pooled = torch.mean(lstm_out, 1)
        return self.classifier(pooled)

model: RoBERTa_BiLSTM = None
tokenizer: RobertaTokenizer = None

def setup_model_files():
    """Ensures the large model is downloaded. Tokenizer files are assumed to be in the root directory."""
    
    # --- 1. HANDLE LARGE MODEL DOWNLOAD ---
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model file already found: {MODEL_PATH}")
        return

    print("❌ Model file not found. Starting download from Google Drive...")
    
    # CRITICAL: This requires 'gdown' to be installed and the Drive link to be public.
    download_command = f"gdown --id {MODEL_FILE_ID} -O {MODEL_PATH}"
    result = os.system(download_command)
    
    if result != 0:
        print("FATAL ERROR: Download command failed! Check if 'gdown' is installed and the Drive link is public.")
        raise RuntimeError("Model download failed.")
    
    print(f"✅ Model file downloaded successfully to {MODEL_PATH}")


def load_model_and_tokenizer() -> Tuple[RoBERTa_BiLSTM, RobertaTokenizer]:
    """Loads the trained model weights and tokenizer."""
    global model, tokenizer
    
    # STEP 1: Download model file if missing
    setup_model_files() 
    
    # STEP 2: Load Tokenizer (It will look in the root directory '.')
    try:
        tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.add_tokens(["_"])
        print(f"✅ Tokenizer loaded from current directory ({TOKENIZER_PATH})")
    except Exception as e:
        # If the tokenizer files are missing from the root directory, this will catch it.
        print(f"FATAL ERROR: Could not load tokenizer from {TOKENIZER_PATH}. Files must be in the same directory as bulldog.py: {e}")
        raise RuntimeError("Tokenizer loading failed.")
        
    # STEP 3: Instantiate and Load Model
    try:
        model = RoBERTa_BiLSTM(num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("✅ Model weights loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"FATAL ERROR: Could not load model from {MODEL_PATH}: {e}")
        raise RuntimeError("Model loading failed after file setup.")

def run_model_prediction(text: str) -> int:
    """Returns the model's prediction (1 for harmful, 0 for non-harmful)."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        try:
            load_model_and_tokenizer() 
        except RuntimeError:
            return 0 

    enc = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=1).item() 
        
    return prediction

# ==============================================================
# 🤖 3. DISCORD BOT IMPLEMENTATION (Core Logic & Slash Commands)
# ==============================================================

intents = discord.Intents.default()
intents.message_content = True 
intents.guilds = True 
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    # --- CRITICAL DEBUGGING STEP ADDED HERE ---
    print("--- STARTING MODEL SETUP ---")

    try:
        load_model_and_tokenizer() 
        
        # Syncing commands will only happen after model is loaded successfully
        print("Attempting command synchronization...")
        await bot.tree.sync()
        print("✅ Commands synced.")
        
        print(f'Bot connected as {bot.user}')
        print('Bulldog Bot is active and monitoring messages.')
    except RuntimeError as e: 
        print(f"FATAL SETUP ERROR: {e}")
        print("Bot failed to start. Shutting down.")
        await bot.close()

# --- Slash Command Group: /set alerts channel ---
set_group = app_commands.Group(name="set", description="Set up various bot configurations.")
bot.tree.add_command(set_group)
alerts_group = set_group.group(name="alerts", description="Configuration for alert messages.")

@alerts_group.command(name="channel", description="Sets the channel for private moderator alerts.")
@app_commands.checks.has_permissions(administrator=True)
async def set_alerts_channel_slash(interaction: discord.Interaction, channel: discord.TextChannel):
    global MOD_ALERT_CHANNEL_ID
    
    if interaction.guild is None:
        return await interaction.response.send_message("This command must be run in a server.", ephemeral=True)

    MOD_ALERT_CHANNEL_ID = channel.id
    
    await interaction.response.send_message(
        f"✅ Moderator alert channel set to: {channel.mention}. Alerts will now be sent here.",
        ephemeral=True
    )

# --- Manual Sync Command ---
@bot.command(name='sync')
@commands.is_owner()
async def sync_command(ctx):
    await ctx.send("Attempting **Global** synchronization... 🌍")
    try:
        synced = await bot.tree.sync()
        await ctx.send(f"✅ **Globally** synced **{len(synced)}** commands. Use `/set alerts channel` now.")
    except Exception as e:
        await ctx.send(f"❌ Global sync failed: ```{e}```")


@bot.event
async def on_message(message):
    global MOD_ALERT_CHANNEL_ID
    
    if message.author == bot.user or message.guild is None:
        return
    
    # This call now includes the model loading and prediction logic
    prediction = run_model_prediction(message.content)
    
    if prediction == 1:
        
        # --- A. Public Safety Message ---
        public_alert = (
            "# Bulldog Alert :siren:\n"
            "Bulldog Bot has detected harmful activity.\n"
            "If you or someone you know is going through a hard time, please reach out to friends, family, and professionals.\n"
            "National Suicide Prevention Lifeline: **988 (US)**\n\n"
            "Please disregard this message if it was a mistake.\n"
            "This bot is not a replacement for professional help."
        )
        await message.channel.send(public_alert)
        
        # --- B. Private Moderator Alert ---
        if MOD_ALERT_CHANNEL_ID is not None:
            mod_channel = bot.get_channel(MOD_ALERT_CHANNEL_ID)
            
            if mod_channel:
                embed = discord.Embed(
                    title="🚨 URGENT: Harmful Content Detected",
                    description="Bulldog Bot has flagged a message for immediate moderator investigation.",
                    color=discord.Color.red()
                )
                embed.add_field(name="User", value=f"{message.author.mention} (`{message.author.id}`)", inline=True)
                embed.add_field(name="Channel", value=f"{message.channel.mention}", inline=True)
                embed.add_field(name="Message Content", value=f"```\n{message.content[:1024]}\n```", inline=False)
                embed.add_field(name="Action Link", value=f"[Jump to Message]({message.jump_url})", inline=False)
                embed.set_footer(text="Please investigate the situation. Disregard if this is a false report. The bot is heavily trained and screw ups should be rare.")

                await mod_channel.send(embed=embed)

    await bot.process_commands(message)

# Start the bot
bot.run(DISCORD_TOKEN)
