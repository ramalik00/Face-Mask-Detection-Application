import datetime
class Fps:
	def __init__(self):
		#stores the start time, end time, and total number of frames
		self._start=None
		self._end=None
		self._numFrames=0
	def start(self):
		#starts the timer
		self._start=datetime.datetime.now()
		return self
	def stop(self):
		#stops the timer
		self._end=datetime.datetime.now()
	def update(self):
		self._numFrames+=1
	def elapsed(self):
		#return the total number of seconds in the process
		return (self._end-self._start).total_seconds()
	def fps(self):
		#calculates frames per second
		return self._numFrames/self.elapsed()
