# bulldog_bot.py
import os
import asyncio
import torch
import torch.nn as nn
import discord
from discord.ext import commands
from transformers import RobertaModel, RobertaTokenizer

# -----------------------------
# Configuration and global state
# -----------------------------
ALERT_CONFIDENCE_THRESHOLD = 0.75  # tune this based on your validation
CHECKPOINT_PATH = "final_checkpoint.pt"  # ensure this file exists locally
ROBERTA_NAME = "roberta-base"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Per-server alert channel mapping
alert_channels = {}

# -----------------------------
# Model definition and utilities
# -----------------------------
class RobertaBiLSTM(nn.Module):
    def __init__(self, roberta_name=ROBERTA_NAME, hidden_size=768, lstm_hidden=256, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(roberta_name)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(enc.last_hidden_state)   # [batch, seq, 2*lstm_hidden]
        pooled = lstm_out[:, -1, :]                      # last timestep pooling
        logits = self.classifier(self.dropout(pooled))   # [batch, num_classes]
        return {"logits": logits}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = RobertaTokenizer.from_pretrained(ROBERTA_NAME)
MODEL = RobertaBiLSTM().to(DEVICE)
MODEL_LOADED = False

def preprocess(text: str):
    batch = TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    return batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)

def classify_text(text: str):
    with torch.no_grad():
        input_ids, attention_mask = preprocess(text)
        out = MODEL(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]                # [1, 2]
        probs = torch.softmax(logits, dim=1)  # [1, 2]
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    return pred, confidence, logits.cpu().tolist()

# -----------------------------
# Discord bot events and commands
# -----------------------------
@bot.event
async def on_ready():
    global MODEL_LOADED
    print("Bulldog starting up...")

    # Give Discord a moment before syncing commands
    await asyncio.sleep(2.0)
    try:
        # For testing, sync to a specific guild for instant availability
        # Replace YOUR_GUILD_ID with your test server's ID
        guild = discord.Object(id=1408670998001094666)
        await bot.tree.sync(guild=guild)
        print("Slash commands synced to test guild.")
    except Exception as e:
        print(f"Command sync failed: {e}")

    # Load model checkpoint once at startup
    if not MODEL_LOADED:
        try:
            state = torch.load(CHECKPOINT_PATH, map_location="cpu")
            MODEL.load_state_dict(state)
            MODEL.eval()
            MODEL_LOADED = True
            print(f"Model loaded from {CHECKPOINT_PATH} on {DEVICE}.")
            # quick sanity inference
            _pred, _conf, _logits = classify_text("this is a quick startup test")
            print(f"Startup test -> pred:{_pred} conf:{_conf:.3f} logits:{_logits}")
        except Exception as e:
            print(f"Failed to load model checkpoint '{CHECKPOINT_PATH}': {e}")

    print(f"{bot.user} is online and ready.")

@bot.tree.command(name="ping", description="Check if Bulldog is alive")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong 🐶")

@bot.tree.command(name="setalerts", description="Set the alerts channel for this server")
async def setalerts(interaction: discord.Interaction, channel: discord.TextChannel):
    alert_channels[interaction.guild.id] = channel.id
    await interaction.response.send_message(
        f"✅ Alerts will now be sent to {channel.mention}", ephemeral=True
    )

@bot.tree.command(name="classify", description="Classify a piece of text with Bulldog's model")
async def classify_cmd(interaction: discord.Interaction, text: str):
    if not MODEL_LOADED:
        await interaction.response.send_message("Model is still loading. Try again in a moment.", ephemeral=True)
        return
    pred, conf, logits = classify_text(text)
    label = "risky" if pred == 1 else "safe"
    await interaction.response.send_message(
        f"Label: {label} | Confidence: {conf:.3f} | Logits: {logits}"
    )

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    if not MODEL_LOADED:
        return

    pred, conf, logits = classify_text(message.content)

    if pred == 1 and conf >= ALERT_CONFIDENCE_THRESHOLD:
        guild_id = message.guild.id if message.guild else None

        if guild_id and guild_id in alert_channels:
            alert_channel = bot.get_channel(alert_channels[guild_id])
            if alert_channel:
                await alert_channel.send(
                    f":siren: NEW ALERT: {message.author.mention}\n"
                    f"Message: {message.content}\n"
                    f"Label: risky | Confidence: {conf:.3f}\n"
                    f"Logits: {logits}"
                )
                return

        try:
            await message.author.send(
                "# :siren: Official Bulldog Alert! Your message was flagged! Please contact the national suicide prevention line: 988. Reach out to family, friends, or seek professional help."
            )
        except Exception:
            pass

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN environment variable not set.")
    bot.run(token)
