import cv2
import os
import numpy as np
from moviepy.editor import *
from abc import ABC, abstractmethod
import subprocess
import audio
import math
import enlighten

from util import smoothstep_map
from settings import Settings

class Video:
    class WrongOpenType(Exception):
        pass
    class NotOpen(Exception):
        pass

    def __init__(self, path):
        self._path = path
        self._audio = None
        self._openedForRead = False
        self._openedForWrite = False
        self._reader = None
        self._writer = None
        self._writtenFrameCount = 0

    def openForRead(self):
        if os.path.exists(self.path):
            self._reader = cv2.VideoCapture(self.path)
            self._openedForRead = True
        else:
            raise FileNotFoundError
    
    def openForWrite(self, fourcc, fps, width, height):
        if not os.path.exists(os.path.dirname(self.path)):
            if len(os.path.dirname(self.path)) > 0:
                os.makedirs(os.path.dirname(self.path))
        self._writer = cv2.VideoWriter(
            self.path,
            fourcc,
            fps,
            (width, height)
        )

    def close(self):
        if self._reader != None:
            self._reader.release()
            self._reader = None
            self._openedForRead = False

        if self._writer != None:
            self._writer.release()
            tempFile = Settings.getTempFile("mp4")
            if self.audio:
                subprocess.run([
                    "ffmpeg", "-y", 
                    "-i", self.path, "-i", self.audio.path, 
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-shortest", tempFile
                ], capture_output=True)
                os.remove(self.path)
                subprocess.run([
                    "ffmpeg", "-y", 
                    "-i", tempFile, "-i", self.audio.path, 
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-shortest", self.path
                ], capture_output=True)
            self._writer = None
            self._openedForWrite = False
            self._writtenFrameCount = 0

    def readFrame(self):
        if self._openedForWrite:
            raise Video.WrongOpenType
        return self._reader.read()

    def writeFrame(self, frame):
        if self._openedForRead:
            raise Video.WrongOpenType
        res = self._writer.write(frame)
        self._writtenFrameCount += 1
        return res

    def copy(self, outputPath):
        def nothing(frame, frameNo, inputVid, outputVid):
            return frame, True
        return Filter.applyToEachFrame(self, outputPath, nothing)

    @property
    def audio(self):
        return self._audio

    @audio.setter
    def audio(self, path):
        self._audio = audio.Audio(path)

    @property
    def path(self):
        return self._path

    @property
    def fps(self):
        if self._openedForRead:
            return int(self._reader.get(cv2.CAP_PROP_FPS))
        elif self._openedForWrite:
            return int(self._writer.get(cv2.CAP_PROP_FPS))

    @property
    def fourcc(self):
        if self._openedForRead:
            return int(self._reader.get(cv2.CAP_PROP_FOURCC))
        elif self._openedForWrite:
            return int(self._writer.get(cv2.CAP_PROP_FOURCC))

    @property
    def width(self):
        if self._openedForRead:
            return int(self._reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        elif self._openedForWrite:
            return int(self._writer.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        if self._openedForRead:
            return int(self._reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self._openedForWrite:
            return int(self._writer.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def framesWritten(self):
        return self._writtenFrameCount

    @property
    def frameCount(self):
        if self._openedForRead:
            return int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def opened(self):
        return self._openedForRead or self._openedForWrite

class Filter(ABC):
    @classmethod
    @abstractmethod
    def help(cls):
        """ A help message for the filter
        """
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def apply(cls, video):
        """ Apply the filter to a video
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def name(cls):
        """ The name of this filter
        """
        raise NotImplementedError
    
    @classmethod
    def applyToEachFrame(cls, video, outputPath, func):
        outputVid = Video(outputPath)
        if video.audio != None:
            outputVid.audio = video.audio.path

        video.openForRead()
        outputVid.openForWrite(video.fourcc, video.fps, video.width, video.height)

        pbar = enlighten.Counter(total=video.frameCount, desc=func.__name__, unit='frame')
        frameNo = 1
        success = True
        while video.opened and success != False and frameNo < video.frameCount:
            frameNo += 1
            success, frame = video.readFrame()
            if success:
                frame, success = func(frame, frameNo, video, outputVid)
                if success:
                    outputVid.writeFrame(frame)
            pbar.update()

        video.close()
        outputVid.close()
        return outputVid

class Cut(Filter):
    @classmethod
    def name(cls):
        return "cut"

    @classmethod
    def help(cls):
        return "Cut the video between two timestaps, in seconds"

    @classmethod
    def apply(cls, video, start=0, end=-1, outputPath=None):
        def cut(frame, frameNo, inputVid, outputVid):
            if int(start)*inputVid.fps <= frameNo <= int(end)*inputVid.fps:
                return frame, True
            elif frameNo < int(start)*inputVid.fps:
                return frame, None
            return frame, False
        if outputPath == None:
            outputPath = Settings.getTempFile("mp4")
        print(f"Cutting {video.path} between {start} and {end} seconds, outputting to {outputPath}")
        res = cls.applyToEachFrame(video, outputPath, cut)
        print("Finished cut")
        return res

class PixelReplace(Filter):
    @classmethod
    def name(cls):
        return "pixelReplace"

    @classmethod
    def help(cls):
        return "Replace a pixel, or range of pixels, with one of another colour"

    @classmethod
    def apply(cls, video, value=0, tolerance=5, outputPath=None):
        def replace(frame, frameNo, inputVid, outputVid):
            if callable(value):
                frame = np.fromfunction(value, frame)
            else:
                mask = (frame < int(tolerance)).all(axis=2)    
                frame[mask] = (value, value, value)
            return frame, True
        value, tolerance = int(value), int(tolerance)
        if outputPath == None:
            outputPath = Settings.getTempFile("mp4")
        print(f"Replacing pixels in the range of {value - tolerance} to {value + tolerance} for {video.path}, outputting to {outputPath}")
        res = cls.applyToEachFrame(video, outputPath, replace)
        print(f"Finished pixel replacement")
        return res

class AudioPixelReplace(Filter):
    @classmethod
    def name(cls):
        return "audioPixelReplace"

    @classmethod
    def help(cls):
        return "Replace a pixel, or range of pixels, with one of another colour, with the hue based on the sound in an audio track"

    @classmethod
    def apply(cls, video, value=0, tolerance=5, outputPath=None):
        def replace(frame, frameNo, inputVid, outputVid):
            pitch, success = inputVid.audio.pitchAtFrame(frameNo, inputVid.fps)
            if success:
                mask = (frame < [tolerance, tolerance, tolerance]).all(axis=2)
                pixelValue = int((np.mean(pitch)/inputVid.audio.max)*255) + int(255/2)
                frame[mask] = (255-pixelValue, 0, int(255/2)-pixelValue)
            return frame, True
        value, tolerance = int(value), int(tolerance)
        if outputPath == None:
            outputPath = Settings.getTempFile("mp4")
        if video.audio != None:
            video.audio.read()
        print(f"Replacing pixels in the range of {value - tolerance} to {value + tolerance} for {video.path}, outputting to {outputPath}")
        res = cls.applyToEachFrame(video, outputPath, replace)
        print(f"Finished pixel replacement")
        return res

class Denoise(Filter):
    @classmethod
    def name(cls):
        return "denoise"

    @classmethod
    def help(cls):
        return "Apply opencv's fastNlMeansDenoising algorithm to each frame"

    @classmethod
    def apply(cls, video, outputPath=None):
        def denoiseFrame(frame, frameNo, inputVid, outputVid):
            return cv2.fastNlMeansDenoising(frame, None), True
        if outputPath == None:
            outputPath = Settings.getTempFile("mp4")
        print(f"Applying denoising algorithm to {video.path}, outputting to {outputPath}")
        res = cls.applyToEachFrame(video, outputPath, denoiseFrame)
        print(f"Finished denoising video")
        return res

class LowPass(Filter):
    @classmethod
    def name(cls):
        return "lowpass"

    @classmethod
    def help(cls):
        return """Low pass all pixels so that any under a certain threshold are set to 0.
Pass the threshold (0-255 integer) below which pixels will be zero'd out."""

    @classmethod
    def apply(cls, video, outputPath=None):
        def lowPass(frame, frameNo, inputVid, outputVid):
            frame[frame < (255 // 10)] = 0
            return frame, True
        if outputPath == None:
            outputPath = Settings.getTempFile("mp4")
        print(f"Applying low pass algorithm to {video.path}, outputting to {outputPath}")
        res = cls.applyToEachFrame(video, outputPath, lowPass)
        print(f"Finished denoising video")
        return res

class DeFlicker(Filter):
    @classmethod
    def name(cls):
        return "deflicker"

    @classmethod
    def help(cls):
        return "Reduce flickering in the video"

    @classmethod
    def apply(cls, video, outputPath=None):
      # could reduce flickering by smoothstepping the changes in color so that they appear more gradually
      pbar = enlighten.Counter(total=video.frameCount, desc=cls.name(), unit='frame')
      if outputPath == None:
          outputPath = Settings.getTempFile("mp4")
      print(f"Applying frame differencing to {video.path}, outputting to {outputPath}")
      video.openForRead()
      outputVid = Video(outputPath)
      if video.audio != None:
          outputVid.audio = video.audio.path
      outputVid.openForWrite(video.fourcc, video.fps, video.width, video.height)
      frameNo = 0
      success, previous = video.readFrame()
      prevStddev = np.std(previous)
      while video.opened and success:
          frameNo += 1
          success, current = video.readFrame()
          if success:
              # frame = np.uint8(smoothstep_map(previous, current, 2))
              frame = np.uint8((current + previous) / 2)
              outputVid.writeFrame(frame)
              previous = current
          pbar.update()
      video.close()
      outputVid.close()
      print("Finished frame differencing")
      return outputVid
      

class FrameDifference(Filter):
    @classmethod
    def name(cls):
        return "frameDifference"

    @classmethod
    def help(cls):
        return "Apply frame differencing to the video"

    @classmethod
    def apply(cls, video, outputPath=None):
      # could reduce flickering by smoothstepping the changes in color so that they appear more gradually
        pbar = enlighten.Counter(total=video.frameCount, desc=cls.name(), unit='frame')
        if outputPath == None:
            outputPath = Settings.getTempFile("mp4")
        print(f"Applying frame differencing to {video.path}, outputting to {outputPath}")
        video.openForRead()
        outputVid = Video(outputPath)
        if video.audio != None:
            outputVid.audio = video.audio.path
        outputVid.openForWrite(video.fourcc, video.fps, video.width, video.height)
        frameNo = 0
        success, previous = video.readFrame()
        prevStddev = np.std(previous)
        while video.opened and success:
            frameNo += 1
            success, current = video.readFrame()
            if success:
                # frame = np.uint8(smoothstep_map(previous, current, 2))
                frame = np.uint8(np.absolute(np.subtract(np.int16(previous), np.int16(current))))
                stdDev = np.std(frame)
                if stdDev < 2 * prevStddev:
                    outputVid.writeFrame(frame)
                prevStddev = stdDev
                outputVid.writeFrame(frame)
                previous = current
            pbar.update()
        video.close()
        outputVid.close()
        print("Finished frame differencing")
        return outputVid