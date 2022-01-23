import vosk
import wave
import json
from pydub import AudioSegment
from datetime import datetime
import os

class transcriber:
	
	def __init__(self):
		vosk.SetLogLevel(-1)
		model_path = 'vosk-model-small-de-0.15'
		#model_path = 'vosk-model-de-0.21'
		spk_model_path = 'vosk-model-spk-0.4'
		sample_rate = 16000
		model = vosk.Model(model_path)
		spk_model = vosk.SpkModel(spk_model_path)
		self.rec = vosk.KaldiRecognizer(model, sample_rate)
		self.rec.SetSpkModel(spk_model)
		self.rec.SetWords(True)
		

	def cosine_dist(self, x, y):
		nx = np.array(x)
		ny = np.array(y)
		return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)
		
	def get_speaker_from_split(self, audio_file, start, stop):
		audio = AudioSegment.from_wav(audio_file)
		excerpt = audio[start*1000:stop*1000]
		timestamp = datetime.now().microsecond
		excerpt.export(str(timestamp) + ".wav", format='wav')
		wf = wave.open(str(timestamp) + ".wav", "rb")
		while True:
		   data = wf.readframes(4000)
		   if len(data) == 0:
			   break
		   if self.rec.AcceptWaveform(data):
			   self.rec.Result()
		res = self.rec.FinalResult()
		jres = json.loads(res)
		wf.close()
		os.remove(str(timestamp) + ".wav")
		if 'spk' in jres:
			return jres['spk']
		else:
			return []
		

	def transcribe(self, audio_file, csv_output):
	
		wf = wave.open(audio_file, "rb")
		
		results = []
		while True:
		   data = wf.readframes(4000)
		   if len(data) == 0:
			   break
		   if self.rec.AcceptWaveform(data):
			   results.append(self.rec.Result())
		results.append(self.rec.FinalResult())

		subs = []
		for i, res in enumerate(results):
			print("Result " + str(i) + ": " + str(res))
			jres = json.loads(res)
			if not 'result' in jres:
				continue
			words = jres['result']
			for j in range(0, len(words)):
				line = words[j] 
				print("Line: " + str(line))
				#s = srt.Subtitle(index=len(subs), 
				#	content=" ".join([l['word'] for l in line]),
				#	start=datetime.timedelta(seconds=line[0]['start']), 
				#	end=datetime.timedelta(seconds=line[-1]['end']))
				#subs.append(s)
				speaker = self.get_speaker_from_split(audio_file, line['start'], line['end'])
				print(line['word'] + " " + str(line['conf']) + " " + str(line['start']) + " " + str(line['end']) + " " + str(speaker))
		return subs
	
if __name__ == '__main__':
	t = transcriber()
	transcription = t.transcribe('meeting.wav', 'out.csv')
	print(transcription)