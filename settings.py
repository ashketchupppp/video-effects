import os
import shutil

class Settings:
    inputFilePath = None
    filetype = None
    outputFilePath = None
    verbose = False
    tempFileCount = 0
    audio = None
    tempDir = ".temp"
    filters = []

    @staticmethod
    def getTempFile(ext):
        if not os.path.exists(Settings.tempDir):
            os.mkdir(Settings.tempDir)
        Settings.tempFileCount += 1
        return Settings.tempDir + os.sep + str(Settings.tempFileCount) + "." + ext

def ApplyDefaults():
    if Settings.outputFilePath == None:
        Settings.outputFilePath  = f"filtered_{Settings.inputFilePath}"