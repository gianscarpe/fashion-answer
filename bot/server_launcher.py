from Updater import Updater
import os, sys, platform, subprocess
from config import BOT_TOKEN
from matcher import match

def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def imageHandler(bot, message, chat_id, local_filename):
    print(local_filename)
    # send message to user
    bot.sendMessage(chat_id, "Hi, please wait until the image is ready")
    # set matlab command
    if 'Linux' in platform.system():
        matlab_cmd = '/usr/local/bin/matlab'
    else:
        matlab_cmd = '"C:\\Program Files\\MATLAB\\R2016a\\bin\\matlab.exe"'

    result = match(local_filename)

    bot.sendImage(chat_id, result, "")


if __name__ == "__main__":
    bot_id = BOT_TOKEN
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.start()
