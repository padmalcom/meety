import vosk
import wave
import json
from pydub import AudioSegment
from datetime import datetime
import os
import numpy as np
import pandas as pd

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
		
		self.speakers={}
		self.speakers["jonas"] = [-0.849626, 0.1394, 0.377193, 0.04026, -0.516814, 0.684988, 0.276028, 0.44519, -0.071336, 0.236369, 1.438427, -1.16203, 0.171904, -0.246187, 1.162354, 1.754254, -0.870339, 0.581138, 0.277063, -0.213094, -0.178992, 1.307812, 0.383807, -1.808675, -0.183495, 0.595374, 0.552995, 1.130693, 1.509223, 0.55946, 0.377448, -0.874153, -0.805022, -0.053774, -0.024277, 0.100899, -0.16061, 0.979247, -0.806801, 1.075389, 0.054593, 0.457873, -0.123776, 0.102562, -1.03972, -0.405821, -2.296902, 1.552534, 1.987395, -0.178972, -1.699092, -1.007383, -0.044273, 0.668422, -0.387606, 1.032058, 1.218006, -0.004968, -2.110302, -0.588267, -1.630525, -0.486805, -0.149978, -0.323995, 0.057739, -1.591965, -0.507674, 2.15651, 0.087731, 0.380945, 1.659711, 1.078468, -0.96696, 1.021768, -0.429574, 1.221862, -1.227425, -0.760518, 0.883041, 1.018671, 1.070466, -0.274897, 0.163634, -0.747442, -0.05772, 0.477834, -2.581385, 0.522382, -1.506052, 1.325543, -0.888436, -0.042877, -0.999906, -0.163342, -0.045787, -2.195781, 1.082597, -0.170146, -0.511741, 0.051157, -1.417514, -1.123549, 0.37979, -0.715364, 0.868087, 0.182235, 1.849722, 2.322327, 0.353243, 0.963822, -1.506676, 1.008916, 0.883645, -0.744136, -0.141038, -0.231468, 0.424568, 0.640757, 1.176899, 0.703505, -1.75687, -0.789799, -0.169159, -1.980155, -0.769657, 0.923687, -0.850422, 1.737328]
		self.new_speaker_index = 0
		
	def get_speaker(self, fingerprint):
		best_speaker = ""
		if len(fingerprint) == 0:
			return best_speaker
		min_cd = 1000
		for s in self.speakers.keys():
			cd = self.cosine_dist(self.speakers[s], fingerprint)
			if cd < min_cd and cd < 0.7:
					best_speaker = s
					min_cd = cd
		if best_speaker == "":
			self.speakers["unknown"+str(self.new_speaker_index)] = fingerprint
			best_speaker = "unknown"+str(self.new_speaker_index)
			self.new_speaker_index += 1
		return best_speaker, min_cd

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
			speaker_name, cd = self.get_speaker(jres['spk'])
			return speaker_name, cd
		else:
			return "", 1000
		

	def transcribe(self, audio_file, csv_output):
	
		speakers = {}
	
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
			jres = json.loads(res)
			if not 'result' in jres:
				continue
			words = jres['result']
			for j in range(0, len(words)):
				line = words[j] 
				speaker, cd = self.get_speaker_from_split(audio_file, line['start'], line['end'])
				subs.append((line['word'], line['conf'], line['start'], line['end'], speaker, cd))
		df = pd.DataFrame.from_records(subs, columns=['word', 'confidence', 'start', 'end', 'speaker', 'cd'])
		df = df.sort_values('start')
		
		# repair sentences
		df.to_csv(csv_output, index=False)
		return subs
	
if __name__ == '__main__':
	t = transcriber()
	transcription = t.transcribe('meeting.wav', 'out.csv')
	print(transcription)