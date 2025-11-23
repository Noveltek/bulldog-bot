import discord
from discord.ext import commands
import torch

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Dictionary to store alert channel per server
alert_channels = {}

# Dummy model loader (replace with your actual model)
def load_model():
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return {"logits": torch.randn(1, 2)}
    return DummyModel()

# --- Slash Commands ---

@bot.event
async def on_ready():
    await bot.tree.sync()   # sync slash commands with Discord
    print(f"{bot.user} is online and commands are synced!")

@bot.tree.command(name="ping", description="Check if Bulldog is alive")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong 🐶")

@bot.tree.command(name="classify", description="Classify a piece of text")
async def classify(interaction: discord.Interaction, text: str):
    mdl = load_model()
    input_ids = torch.randint(0, 100, (1, 10))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = mdl(input_ids=input_ids, attention_mask=attention_mask)
    await interaction.response.send_message(f"Model output: {output}")

@bot.tree.command(name="setalerts", description="Set the alerts channel for this server")
async def setalerts(interaction: discord.Interaction, channel: discord.TextChannel):
    alert_channels[interaction.guild.id] = channel.id
    await interaction.response.send_message(
        f"✅ Alerts will now be sent to {channel.mention}", ephemeral=True
    )

# --- Automatic Monitoring ---

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content.lower()
    risky_phrases = ["i wanna die", "you deserve to die", "kill myself", "hate you"]
    is_risky = any(phrase in content for phrase in risky_phrases)

    if is_risky:
        mdl = load_model()
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = mdl(input_ids=input_ids, attention_mask=attention_mask)

        guild_id = message.guild.id
        if guild_id in alert_channels:
            alert_channel = bot.get_channel(alert_channels[guild_id])
            if alert_channel:
                await alert_channel.send(
                    f"🚨 Offensive message detected:\n"
                    f"User: {message.author.mention}\n"
                    f"Message: {message.content}\n"
                    f"Model output: {output}"
                )
        else:
            # fallback: DM the author
            try:
                await message.author.send("⚠️ Your message was flagged as harmful.")
            except:
                pass

    await bot.process_commands(message)

# Run your bot
bot.run("YOUR_BOT_TOKEN")
