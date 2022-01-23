import vosk
import librosa
import numpy
import os
import math
import json
import pandas

# https://stackoverflow.com/questions/64153590/audio-signal-split-at-word-level-boundary

def extract_text(res):
	jres = json.loads(res)
	text = ''
	spk = ''
	print(jres)
	if  'partial' in jres:
		text = jres['partial']
	if 'spk' in jres:
		spk = jres['spk']
	return text, spk

def transcribe_words(recognizer, bytes):
	results = []

	chunk_size = 4000
	for chunk_no in range(math.ceil(len(bytes)/chunk_size)):
		start = chunk_no*chunk_size
		end = min(len(bytes), (chunk_no+1)*chunk_size)
		data = bytes[start:end]

		if recognizer.AcceptWaveform(data):
			#words = extract_words(recognizer.Result())
			#results += words
			print(recognizer.Result())
		else:
			text, spk = extract_text(recognizer.PartialResult())
			results.append((start, end, text, spk))
			print(results[len(results)-1])
		
	print(recognizer.FinalResult())
	#results += extract_words(recognizer.FinalResult())

	return results

def cosine_dist(x, y):
	nx = np.array(x)
	ny = np.array(y)
	return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)

if __name__ == '__main__':
	vosk.SetLogLevel(-1)

	audio_path = "meeting.wav"
	out_path = "out.csv"

	model_path = 'vosk-model-small-de-0.15'
	spk_model_path = 'vosk-model-spk-0.4'
	sample_rate = 16000

	audio, sr = librosa.load(audio_path, sr=16000)

	# convert to 16bit signed PCM, as expected by VOSK
	int16 = numpy.int16(audio * 32768 / 2).tobytes()

	model = vosk.Model(model_path)
	spk_model = vosk.SpkModel(spk_model_path)
	recognizer = vosk.KaldiRecognizer(model, sample_rate)
	recognizer.SetSpkModel(spk_model)

	res = transcribe_words(recognizer, int16)
	df = pandas.DataFrame.from_records(res)
	#df = df.sort_values('start')

	df.to_csv(out_path, index=False)
	print('Word segments saved to', out_path)

