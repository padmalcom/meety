import vosk
import sys
import os
import wave
import json
import datetime

WORDS_PER_LINE = 7

def transcribe():

	vosk.SetLogLevel(-1)

	audio_path = "meeting.wav"
	out_path = "out.csv"

	model_path = 'vosk-model-small-de-0.15'
	#model_path = 'vosk-model-de-0.21'
	spk_model_path = 'vosk-model-spk-0.4'
	sample_rate = 16000
	
	model = vosk.Model(model_path)
	spk_model = vosk.SpkModel(spk_model_path)
	rec = vosk.KaldiRecognizer(model, sample_rate)
	rec.SetSpkModel(spk_model)
	rec.SetWords(True)
	
	wf = wave.open(audio_path, "rb")
	
	results = []
	subs = []
	while True:
	   data = wf.readframes(4000)
	   if len(data) == 0:
		   break
	   if rec.AcceptWaveform(data):
		   results.append(rec.Result())
	results.append(rec.FinalResult())

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
	return subs
	
if __name__ == '__main__':
	print(transcribe())