import discord
from discord.ext import commands
import json
import os
import torch

# --- Load your model ---
# Make sure roberta_bilstm_final.pt is in the same folder
model = torch.nn.Module()  # placeholder for your actual model class
model.load_state_dict(torch.load("roberta_bilstm_final.pt", map_location="cpu"))

# --- Discord setup ---
intents = discord.Intents.default()
intents.message_content = True  # required for reading messages
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Channel persistence ---
CHANNEL_FILE = "channels.json"
alert_channels = {}

def save_channels():
    with open(CHANNEL_FILE, "w") as f:
        json.dump(alert_channels, f)

def load_channels():
    global alert_channels
    try:
        with open(CHANNEL_FILE, "r") as f:
            alert_channels = json.load(f)
    except FileNotFoundError:
        alert_channels = {}

load_channels()

# --- Command: setalertchannel (admins only) ---
@bot.command(name="setalertchannel")
@commands.has_permissions(administrator=True)
async def setalertchannel(ctx, channel: discord.TextChannel):
    alert_channels[str(ctx.guild.id)] = channel.id
    save_channels()
    await ctx.send(f"✅ Bulldog alerts will now go to {channel.mention}")

@setalertchannel.error
async def setalertchannel_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ Only administrators can set the alert channel.")

# --- DM message template ---
DM_MESSAGE = (
    "# Bulldog Alert 🚨\n"
    "Your recent activity shows that you are going through a rough time.\n"
    "You are not alone. Please consider reaching out to a trusted friend, "
    "family member, or a professional who can support you.\n"
    "Please call the National Suicide Prevention Lifeline: **988**."
)

# --- Event: on_message ---
@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Example: run your model here to classify risk
    # Replace with your actual inference code
    text = message.content
    risk_detected = "suicide" in text.lower()  # placeholder logic

    if risk_detected:
        try:
            # DM the user
            await message.author.send(DM_MESSAGE)
        except discord.Forbidden:
            pass  # user has DMs closed

        # Alert moderators in the chosen channel
        channel_id = alert_channels.get(str(message.guild.id))
        if channel_id:
            channel = bot.get_channel(channel_id)
            if channel:
                await channel.send(f"🚨 Risk detected in {message.author.mention}'s message:\n> {message.content}")

    # Allow commands to still work
    await bot.process_commands(message)

# --- Run the bot ---
bot.run(os.getenv("DISCORD_TOKEN"))
