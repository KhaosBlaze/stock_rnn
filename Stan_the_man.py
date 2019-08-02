import discord
from discord.ext import commands
from discord.ext.commands import Bot
import asyncio
import os
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
async def is_stanley_awake():
	lil_stan = ''
	big_stan = ''
	lil_stan = os.popen('pgrep -f "stanley.py" > /dev/null && echo Running').read()
	big_stan = os.popen('pgrep -f "Stanley.py" > /dev/null && echo Running').read()
	if lii_stan == 'Running' or big_stan == 'Running':
		await bot.say("Stanley still chuggin!")

@bot.command(pass_context=True)
async def sort_me(ctx):
	await bot.say("You're wanting me to sort you into a fictitious house? From a fictitious world? What do I look like, some Sorting hat? Here's your damn hat")
	await bot.say("https://vignette.wikia.nocookie.net/harrypotter/images/6/62/Sorting_Hat.png/revision/latest?cb=20161120072849")

bot.run("NDY2MTE4MDU3ODQwODY5Mzc2.DiXk5g.gOzcGyDg8W__fizmj1pBmrXQzVA")