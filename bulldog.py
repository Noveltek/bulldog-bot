import discord
from discord.ext import commands

# --- Intents ---
intents = discord.Intents.default()
intents.message_content = True  # Required for on_message and prefix commands

# --- Bot Setup ---
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Prefix Command Example (!ping) ---
@bot.command(name="ping")
async def ping_command(ctx):
    await ctx.send("Pong! 🐶")

# --- Slash Command Example (/ping) ---
@bot.tree.command(name="ping", description="Check if Bulldog is alive")
async def ping_slash(interaction: discord.Interaction):
    await interaction.response.send_message("Pong from slash! 🐶")

# --- Slash Command Example (/to) ---
@bot.tree.command(name="to", description="Test slash command")
async def to_command(interaction: discord.Interaction):
    await interaction.response.send_message("Slash command works! 🐾")

# --- On Ready ---
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    for guild in bot.guilds:
        print("In guild:", guild.name, guild.id)
    try:
        guild_id = YOUR_GUILD_ID  # replace with your server ID
        synced = await bot.tree.sync(guild=discord.Object(id=guild_id))
        print(f"Synced {len(synced)} commands to guild {guild_id}")
    except Exception as e:
        print(f"Sync failed: {e}")

# --- Message Classification ---
@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Run your ML model here
    prediction = classify_message(message.content)  # <-- your function

    # Example response based on classification
    if prediction == "toxic":
        await message.channel.send(f"⚠️ Bulldog detected toxic content: {message.content}")

    # Allow prefix commands to still work
    await bot.process_commands(message)

# --- Dummy classifier for demo ---
def classify_message(text: str) -> str:
    # Replace with your ML model logic
    if "badword" in text.lower():
        return "toxic"
    return "clean"

# --- Run Bot ---
bot.run("YOUR_BOT_TOKEN")
