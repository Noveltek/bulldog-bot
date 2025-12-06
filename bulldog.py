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

# ➡️ 2. TOKENIZER PATH: Set to the root directory ('.')
TOKENIZER_PATH = "." 

# 🚨 3. Model File Details 🚨
MODEL_PATH = "roberta_bilstm_final.pt"

# 4. Target Guild for IMMEDIATE Command Sync (Your Server ID: 1408670998001094666)
TARGET_GUILD_ID = 1408670998001094666
TARGET_GUILD = discord.Object(id=TARGET_GUILD_ID)

# Model Constants
MAX_SEQ_LEN = 256
# Use CPU as your host environment log showed CPU-only torch
DEVICE = torch.device("cpu") 

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

# --- FILE REASSEMBLY LOGIC HAS BEEN REMOVED FOR FINAL CODE ---

def setup_model_files():
    """Confirms the model file is present after manual SFTP upload."""
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: The required model file '{MODEL_PATH}' is missing.")
        print("Please ensure you have manually uploaded the complete 500MB+ file via SFTP.")
        raise RuntimeError("Model file missing after manual upload.")
    print(f"✅ Model file confirmed: {MODEL_PATH}")


def load_model_and_tokenizer() -> Tuple[RoBERTa_BiLSTM, RobertaTokenizer]:
    """Loads the trained model weights and tokenizer."""
    global model, tokenizer
    
    print("--- STEP 1: MODEL FILE CONFIRMATION ---")
    setup_model_files() 
    
    print("--- STEP 2: TOKENIZER LOADING ---")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.add_tokens(["_"])
        print(f"✅ Tokenizer loaded from current directory ({TOKENIZER_PATH})")
    except Exception as e:
        print(f"FATAL ERROR: Could not load tokenizer from {TOKENIZER_PATH}. Files must be in the same directory as bulldog.py: {e}")
        raise RuntimeError("Tokenizer loading failed.")
        
    print("--- STEP 3: MODEL WEIGHTS LOADING ---")
    try:
        model = RoBERTa_BiLSTM(num_classes=2)
        # Loads the weights directly from the single, large file.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("✅ Model weights loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"FATAL ERROR: Could not load model from {MODEL_PATH}: {e}")
        raise RuntimeError("Model loading failed.")

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
# 🤖 3. DISCORD BOT IMPLEMENTATION (Core Logic & Syncing)
# ==============================================================

intents = discord.Intents.default()
intents.message_content = True 
intents.guilds = True 
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print("--- STARTING MODEL SETUP ---")

    try:
        load_model_and_tokenizer() 
        
        # --- COMMAND SYNCHRONIZATION ---
        
        # 1. IMMEDIATE GUILD SYNC (For your specific server)
        print(f"Attempting command sync for target guild: {TARGET_GUILD_ID}...")
        await bot.tree.sync(guild=TARGET_GUILD)
        print("✅ Target guild commands synced successfully. Commands should be available NOW in this server.")
        
        # 2. GLOBAL SYNC (This takes 30-60 minutes)
        print("Attempting GLOBAL command sync (This can take up to an hour)...")
        await bot.tree.sync()
        print("✅ GLOBAL sync initiated. Commands will appear in other servers shortly.")
        
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
        synced_global = await bot.tree.sync()
        synced_guild = await bot.tree.sync(guild=ctx.guild)
        
        await ctx.send(f"✅ **Globally** synced **{len(synced_global)}** commands. Also synced **{len(synced_guild)}** commands immediately to this guild. Use `/set alerts channel` now.")
    except Exception as e:
        await ctx.send(f"❌ Global sync failed: ```{e}```")


@bot.event
async def on_message(message):
    global MOD_ALERT_CHANNEL_ID
    
    if message.author == bot.user or message.guild is None:
        return
    
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
