import discord
from discord.ext import commands

# --- Intents ---
intents = discord.Intents.default()
intents.message_content = True  # Required for on_message and prefix commands

# --- Bot Setup ---
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Slash Commands (MUST be defined BEFORE on_ready) ---
@bot.tree.command(name="ping", description="Check if Bulldog is alive")
async def ping_slash(interaction: discord.Interaction):
    await interaction.response.send_message("Pong from slash! 🐶")

@bot.tree.command(name="to", description="Test slash command")
async def to_command(interaction: discord.Interaction):
    await interaction.response.send_message("Slash command works! 🐾")

# --- Prefix Command (!ping) ---
@bot.command(name="ping")
async def ping_command(ctx):
    await ctx.send("Pong! 🐶")

# --- Message Classification ---
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    prediction = classify_message(message.content)  # Your hybrid model here

    if prediction == "toxic":
        await message.channel.send(f"⚠️ Bulldog detected toxic content: {message.content}")

    await bot.process_commands(message)  # Keep prefix commands working

# --- Dummy Classifier (Replace with your real model) ---
def classify_message(text: str) -> str:
    # Replace this with your actual ML model logic
    if "badword" in text.lower():
        return "toxic"
    return "clean"

# --- On Ready: Sync Commands (Guild + Global) ---
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    for guild in bot.guilds:
        print("In guild:", guild.name, guild.id)

    try:
        guild_id = 1408670998001094666  # 🔁 Replace with your actual server ID (integer)
        guild_obj = discord.Object(id=guild_id)

        # Clear and re-sync to guild for instant visibility
        bot.tree.clear_commands(guild=guild_obj)
        guild_synced = await bot.tree.sync(guild=guild_obj)
        print(f"✅ Synced {len(guild_synced)} commands to guild {guild_id}")

        # Global sync (may take up to 1 hour to propagate)
        global_synced = await bot.tree.sync()
        print(f"🌍 Globally synced {len(global_synced)} commands")

    except Exception as e:
        print(f"❌ Sync failed: {e}")

# --- Run Bot ---
bot.run("YOUR_BOT_TOKEN")  # 🔁 Replace with your actual bot token
