import discord
from discord.ext import commands
from discord.ext.commands import Bot
import asyncio
from random import randint

bot = commands.Bot(command_prefix='$')


@bot.event
async def on_ready():
	print("Ready for insults!")

@bot.command(pass_context=True)
async def ping(ctx):
	await bot.say(":ping_pong: ping!!")
	print("user has pinged")

@bot.command(pass_context=True)
async def user(ctx):
	await bot.say(ctx.message.author)

@bot.command(pass_context=True)
async def miss_adam(ctx):
	if str(ctx.message.author).lower() == "wulfieh#0806":
		await bot.say("Sorted to HufflePuff!")
	elif str(ctx.message.author).lower() == "khaosblaze#2647":
		await bot.say("You finally got something right")
	elif str(ctx.message.author).lower() == "mat#2851":
		await bot.say(":upside_down: Can you hear us better if we're orientated your way?")
	elif str(ctx.message.author).lower() == "donahue#7228":
		await bot.say("Hey Adam, I'm going to specifically "+violence()+" you in your "+body())
	await bot.say("Hey " + best_bud().capitalize() +", I'm gonna " + violence() + " you in your "+body())
	print("Adam imitated")

@bot.command(pass_context=True)
async def sort_me(ctx):
	await bot.say("You're wanting me to sort you into a fictitious house? From a fictitious world? What do I look like, some Sorting hat? Here's your damn hat")
	await bot.say("https://vignette.wikia.nocookie.net/harrypotter/images/6/62/Sorting_Hat.png/revision/latest?cb=20161120072849")

bot.run("NDY2MTE4MDU3ODQwODY5Mzc2.DiXk5g.gOzcGyDg8W__fizmj1pBmrXQzVA")