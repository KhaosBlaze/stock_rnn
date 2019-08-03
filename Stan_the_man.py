import discord
from discord.ext import commands
from discord.ext.commands import Bot
import asyncio
import psutil
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
	for proc in psutil.process_iter():
		if proc.name().lower() == 'python' and 'stanley.py' in proc.cmdline():
			await bot.say("Ya boi still chuggin")


bot.run("NDY2MTE4MDU3ODQwODY5Mzc2.DiXk5g.gOzcGyDg8W__fizmj1pBmrXQzVA")
