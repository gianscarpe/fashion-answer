#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
from config import BOT_TOKEN

# !/usr/bin/python3
from telegram.ext import Updater
from telegram.ext import CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


############################### Bot ############################################
def start(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        reply_markup=main_menu_keyboard(),
        text="I'm a bot, please talk to me!",
    )


def main_menu(bot, update):
    query = update.callback_query
    bot.edit_message_text(
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        text=main_menu_message(),
        reply_markup=main_menu_keyboard(),
    )


def first_menu(bot, update):
    query = update.callback_query
    bot.edit_message_text(
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        text=first_menu_message(),
        reply_markup=first_menu_keyboard(),
    )


def second_menu(bot, update):
    query = update.callback_query
    bot.edit_message_text(
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        text=second_menu_message(),
        reply_markup=second_menu_keyboard(),
    )


# and so on for every callback_data option
def first_submenu(bot, update):
    pass


def second_submenu(bot, update):
    pass


############################ Keyboards #########################################
def main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("MainCategory", callback_data="m1")],
        [InlineKeyboardButton("SubCategory", callback_data="m2")],
        [InlineKeyboardButton("Season", callback_data="m3")],
        [InlineKeyboardButton("Season", callback_data="m3")],
    ]
    return InlineKeyboardMarkup(keyboard)


def first_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("k=1", callback_data="m1_1")],
        [InlineKeyboardButton("k=3", callback_data="m1_2")],
        [InlineKeyboardButton("k=5", callback_data="main")],
    ]
    return InlineKeyboardMarkup(keyboard)


def second_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("Submenu 2-1", callback_data="m2_1")],
        [InlineKeyboardButton("Submenu 2-2", callback_data="m2_2")],
        [InlineKeyboardButton("Main menu", callback_data="main")],
    ]
    return InlineKeyboardMarkup(keyboard)


############################# Messages #########################################
def main_menu_message():
    return "Choose the option in main menu:"


def first_menu_message():
    return "Choose the submenu in first menu:"


def second_menu_message():
    return "Choose the submenu in second menu:"


############################# Handlers #########################################
updater = Updater(BOT_TOKEN)

updater.dispatcher.add_handler(CommandHandler("start", start))
updater.dispatcher.add_handler(CallbackQueryHandler(main_menu, pattern="main"))
updater.dispatcher.add_handler(CallbackQueryHandler(first_menu, pattern="m1"))
updater.dispatcher.add_handler(CallbackQueryHandler(second_menu, pattern="m2"))
updater.dispatcher.add_handler(CallbackQueryHandler(first_submenu, pattern="m1_1"))
updater.dispatcher.add_handler(CallbackQueryHandler(second_submenu, pattern="m2_1"))

updater.start_polling()
