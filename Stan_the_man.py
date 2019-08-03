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
async def user(ctx):
	await bot.say(ctx.message.author)

@bot.command(pass_context=True)
async def done_yet():
	for proc in psutil.process_iter():
		if proc.name().lower() == 'python' and 'stanley.py' in proc.cmdline():
			await bot.say("https://media1.tenor.com/images/a4f8ddfe01f388d1abc7022150398bc8/tenor.gif?itemid=4535581")


@bot.command(pass_context=True)
async def results():
	checker = []
	with open("test.out","r") as results:
		for line in results:
			checker.append(line)

	if all(x == checker[0] for x in chcker):
		await bot.say("https://media1.tenor.com/images/f89c189082d675ca5d27eb5028969beb/tenor.gif?itemid=10555880")
	else:
		await bot.say("https://i.redd.it/usiqjl0cbyr11.jpg")



bot.run("NDY2MTE4MDU3ODQwODY5Mzc2.DiXk5g.gOzcGyDg8W__fizmj1pBmrXQzVA")
