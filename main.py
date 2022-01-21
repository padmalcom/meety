import vosk
import librosa
import numpy
import os
import math
import json
import pandas

# https://stackoverflow.com/questions/64153590/audio-signal-split-at-word-level-boundary

def extract_words(res):
	jres = json.loads(res)
	if not 'text' in jres:
		return []
	words = jres['text']
	return words

def transcribe_words(recognizer, bytes):
	results = []

	chunk_size = 4000
	for chunk_no in range(math.ceil(len(bytes)/chunk_size)):
		start = chunk_no*chunk_size
		end = min(len(bytes), (chunk_no+1)*chunk_size)
		data = bytes[start:end]
		
		print(start)
		print(end)

		if recognizer.AcceptWaveform(data):
			words = extract_words(recognizer.Result())
			results += words
			print(recognizer.Result())
		else:
			print(recognizer.PartialResult())
		
	print(recognizer.FinalResult())
	results += extract_words(recognizer.FinalResult())

	return results



if __name__ == '__main__':
	vosk.SetLogLevel(-1)

	audio_path = "meeting.wav"
	out_path = "out.csv"

	model_path = 'vosk-model-small-de-0.15'
	sample_rate = 16000

	audio, sr = librosa.load(audio_path, sr=16000)

	# convert to 16bit signed PCM, as expected by VOSK
	int16 = numpy.int16(audio * 32768).tobytes()

	model = vosk.Model(model_path)
	recognizer = vosk.KaldiRecognizer(model, sample_rate)

	res = transcribe_words(recognizer, int16)
	df = pandas.DataFrame.from_records(res)
	#df = df.sort_values('start')

	df.to_csv(out_path, index=False)
	print('Word segments saved to', out_path)

