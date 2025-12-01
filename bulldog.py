import os
import discord
from discord.ext import commands
from discord import app_commands # Required for clean slash commands
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from typing import Tuple

# ==============================================================
# 🛠️ 1. CONFIGURATION: WHAT TO CHANGE AND UPLOAD
# ==============================================================

# 🚨 IMPORTANT: Replace this placeholder token.
DISCORD_TOKEN = "YOUR_BOT_DISCORD_TOKEN_HERE" 

# ➡️ Upload these files/folders to the same directory on your host:
# The file containing your model weights.
MODEL_PATH = "roberta_bilstm_final.pt" 
# The folder containing your tokenizer files (vocab.json, merges.txt, etc.).
TOKENIZER_PATH = "roberta_bilstm_tokenizer"

# Model Constants
MAX_SEQ_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable to store the ID set by the /set alerts channel command
MOD_ALERT_CHANNEL_ID: int = None

# ==============================================================
# 🧠 2. MODEL ARCHITECTURE & LOADING 
# (Based on the structure from thefinalmodel.py)
# ==============================================================

class RoBERTa_BiLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load the base model. This may download files on the first run.
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        # Bi-LSTM layer
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        # Classifier layer (512 comes from 256*2 for bidirectional)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        # Global Average Pooling
        pooled = torch.mean(lstm_out, 1)
        return self.classifier(pooled)

# Global variables to hold the loaded model and tokenizer
model: RoBERTa_BiLSTM = None
tokenizer: RobertaTokenizer = None

def load_model_and_tokenizer() -> Tuple[RoBERTa_BiLSTM, RobertaTokenizer]:
    """Loads the trained model weights and tokenizer from disk."""
    global model, tokenizer
    
    # 1. Load Tokenizer
    try:
        tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.add_tokens(["_"])
    except Exception as e:
        print(f"ERROR: Could not load tokenizer from {TOKENIZER_PATH}. Using fallback tokenizer.")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        tokenizer.add_tokens(["_"])
        
    # 2. Instantiate and Load Model
    try:
        model = RoBERTa_BiLSTM(num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set to evaluation mode
        print(f"✅ Model successfully loaded from {MODEL_PATH}.")
        return model, tokenizer
    except Exception as e:
        print(f"FATAL ERROR: Could not load model from {MODEL_PATH}.")
        print(f"Details: {e}")
        # Terminate if model load fails 
        raise RuntimeError("Model loading failed. Check file paths and upload.")

def run_model_prediction(text: str) -> int:
    """Tokenizes text and returns the model's prediction (1 for suicide, 0 for non-suicide)."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        try:
            # Attempt to load if not ready
            load_model_and_tokenizer() 
        except RuntimeError:
            return 0 # Safely return 0 if model is non-functional

    # Tokenize the text and send to the device
    enc = tokenizer(
        text, 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_SEQ_LEN, 
        return_tensors="pt"
    )
    
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=1).item() 
        
    return prediction # Returns 1 (suicide) or 0 (non-suicide)

# ==============================================================
# 🤖 3. DISCORD BOT IMPLEMENTATION (Core Logic & Commands)
# ==============================================================

# Required intents for reading message content and commands
intents = discord.Intents.default()
intents.message_content = True 
intents.guilds = True 
# We use a non-slash command prefix (!) just for the manual sync command.
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    # Load model once on bot startup
    try:
        load_model_and_tokenizer()
        print(f'Bot connected as {bot.user}')
        print('Bulldog Bot is active and monitoring messages.')
    except RuntimeError:
        print("Bot failed to start due to model loading error. Closing bot.")
        await bot.close()

# --- Slash Command Group: /set alerts channel ---
# 1. Create the top-level command: /set
set_group = app_commands.Group(name="set", description="Set up various bot configurations.")
bot.tree.add_command(set_group)

# 2. Create the subcommand group: /set alerts
alerts_group = set_group.group(name="alerts", description="Configuration for alert messages.")

@alerts_group.command(name="channel", description="Sets the channel for private moderator alerts.")
@app_commands.checks.has_permissions(administrator=True) # Only administrators can use this
async def set_alerts_channel_slash(interaction: discord.Interaction, channel: discord.TextChannel):
    """
    Handles the slash command: /set alerts channel #channel-name
    """
    global MOD_ALERT_CHANNEL_ID
    
    if interaction.guild is None:
        return await interaction.response.send_message("This command must be run in a server.", ephemeral=True)

    MOD_ALERT_CHANNEL_ID = channel.id
    
    # Respond to the user with confirmation
    await interaction.response.send_message(
        f"✅ Moderator alert channel set to: {channel.mention}. Alerts will now be sent here.",
        ephemeral=True
    )
    print(f"MOD_ALERT_CHANNEL_ID set to {MOD_ALERT_CHANNEL_ID} by {interaction.user}.")

# --- Manual Sync Command (For initial slash command setup) ---
@bot.command(name='sync')
@commands.is_owner() # Ensures only the bot owner can run this (set in Discord App Portal)
async def sync_command(ctx):
    """Syncs slash commands globally (needed after first launch or command changes)."""
    await ctx.send("Attempting **Global** synchronization... 🌍")
    try:
        synced = await bot.tree.sync()
        await ctx.send(f"✅ **Globally** synced **{len(synced)}** commands. Use `/set alerts channel` now.")
    except Exception as e:
        await ctx.send(f"❌ Global sync failed: ```{e}```")


@bot.event
async def on_message(message):
    global MOD_ALERT_CHANNEL_ID
    
    # 1. Ignore the bot's own messages and DMs
    if message.author == bot.user or message.guild is None:
        return
    
    # 2. Run the content through the prediction model
    prediction = run_model_prediction(message.content)
    
    # 3. If harmful content is detected (prediction == 1)
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

                embed.add_field(
                    name="User", 
                    value=f"{message.author.mention} (`{message.author.id}`)", 
                    inline=True
                )
                embed.add_field(
                    name="Channel", 
                    value=f"{message.channel.mention}", 
                    inline=True
                )
                embed.add_field(
                    name="Message Content", 
                    value=f"```\n{message.content[:1024]}\n```", 
                    inline=False
                )
                embed.add_field(
                    name="Action Link",
                    value=f"[Jump to Message]({message.jump_url})",
                    inline=False
                )
                embed.set_footer(text="Please investigate the situation. Disregard if this is a false report. The bot is heavily trained and screw ups should be rare.")

                # Send the alert
                await mod_channel.send(embed=embed)
            else:
                print(f"Error: Could not find MOD_ALERT_CHANNEL with ID {MOD_ALERT_CHANNEL_ID}. Has the channel been deleted?")
        else:
            print("Alert Channel not set. Prompting user to set it...")
            # Optionally send a private reminder to the admin who sent the message
            # This is complex to do reliably, so we'll stick to the console print for simplicity.

    # 4. Allow the bot to process other commands
    await bot.process_commands(message)

# Start the bot
bot.run(DISCORD_TOKEN)
