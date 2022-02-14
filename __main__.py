import os
import argparse
from settings import Settings, ApplyDefaults
import cv2
import video
import audio
import sys

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


description = """A command-line tool for applying different visual effects to images and videos.
"""

help = {
    "inputFilePath" : "An input file path.",
    "verbose" : "Enable verbose progress output",
    "outputFilePath" : "The file path to output to",
    "order" : "Specify the order that filters should be applied, comma separated",
    "audio" : "add an audio track to the video"
}

if __name__ == "__main__":
    errorMessages = []

    supportedVideoFileTypes = [
        "mp4"
    ]
    supportedVideoFilters = {f.name() : f for f in video.Filter.__subclasses__()}

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(nargs='?', dest="inputFilePath", type=str, help=help["inputFilePath"])
    parser.add_argument("--verbose", help=help["verbose"], action="store_true")
    parser.add_argument("--outputFilePath", help=help["outputFilePath"])
    parser.add_argument("--order", type=str, help=help["order"])
    parser.add_argument("--audio", type=str, help=help["audio"])

    for filterName in supportedVideoFilters:
        parser.add_argument(f"--{filterName}", help=supportedVideoFilters[filterName].help(), nargs="*", action=ParseKwargs)

    args = parser.parse_args()

    Settings.inputFilePath = args.inputFilePath if args.inputFilePath != None else ""
    Settings.outputFilePath = args.outputFilePath
    Settings.verbose = args.verbose
    Settings.audio = args.audio

    if not os.path.isfile(Settings.inputFilePath):
        errorMessages.append(f"{Settings.inputFilePath} is not a file")

    order = args.order.split(",") if args.order != None else [key for key in supportedVideoFilters]
    for filterName in order:
        if not filterName in order:
            errorMessages.append(f"{filterName} is not a filter")

    for e in errorMessages:
        print(e)
    if len(errorMessages):
        exit(1)

    ApplyDefaults()
    filters = {}
    for filterName in supportedVideoFilters:
        filterValues = eval(f"args.{filterName}")
        if filterValues != None:
            filters[filterName] = filterValues

    current = video.Video(Settings.inputFilePath)
    if Settings.audio != None:
        current.audio = Settings.audio
    for filter in filters:
        current = supportedVideoFilters[filter].apply(current, **filters[filter])

    final = current.copy(Settings.outputFilePath)

    cv2.destroyAllWindows()