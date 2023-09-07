import os
from pydub import AudioSegment
import pathlib
from openai.embeddings_utils import cosine_similarity, get_embedding
import openai
import spacy
from spacy.lang.es import Spanish
from os import listdir
import soundfile as sf
from os.path import isfile, join

def label_score(review_embedding, label_embeddings):
   return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

#initiate spacy with spanish language model
nlp = spacy.load("es_core_news_sm")

#convert text to chunks of 1000 words or less
def text_to_chunks(text):
  chunks = [[]]
  chunk_total_words = 0

  sentences = nlp(text)

  for sentence in sentences.sents:
    chunk_total_words += len(sentence.text.split(" "))

    if chunk_total_words > 1300:
      chunks.append([])
      chunk_total_words = len(sentence.text.split(" "))

    chunks[len(chunks)-1].append(sentence.text)
  
  return chunks

#summarize text in no more than 3 sentences
def summarize_text(text):
    #prompt = f"The follwing text is a summary of a phone conversation between a customer and an account manager. Summarize the text in 4 sentences or less and classify the call into either a complaint or not complaint:\n{text}"
    #prompt = f"Resume el siguiente texto en 5 frases o menos:\n{text}"
    prompt = f"Summarize the following text in 5 sentences or less:\n{text}"

    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt,
        temperature=0.3, 
        max_tokens=250, # = 112 words
        top_p=1, 
        frequency_penalty=0,
        presence_penalty=1
    )
    return response["choices"][0]["text"]

#summarize text in no more than 3 sentences
def classify_sentiment(text):
  prompt = f"Answering with a single word (satisfied or unsatisfied) indicate if either of the people in the text were not satisfied with a service received at any point in the story:\n{text}"

  response = openai.Completion.create(
      engine="text-davinci-003", 
      prompt=prompt,
      temperature=0.3, 
      max_tokens=25, # = 112 words
      top_p=1, 
      frequency_penalty=0,
      presence_penalty=1
  )

  return response["choices"][0]["text"]

#transcribe audio file
def get_transcript(file):
    audio_file= open(file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    file = open(file.replace(".mp3", "_transcript.txt"), "w")
    file.write(transcript.text)
    file.close()

def convert_wav_to_mp3(wav_file, mp3_file):
  #wav = open(wav_file, "rb")
  #print("##############################################")
  #print(wav)
  #print("##############################################")
  #print(mp3_file)
  #print("##############################################")
  try:
    sound = AudioSegment.from_wav(wav_file)
    sound.export(mp3_file, format="mp3")
  except:
    print("Error converting file: " + wav_file)

def resave_all_wav_files(files):
  for file in files:
    data, samplerate = sf.read(file)
    sf.write(file, data, samplerate)

def convert_all_wav_to_mp3(files):
  for file in files:
    convert_wav_to_mp3(file, file.replace(".WAV",".mp3").replace(".wav",".mp3"))

def get_all_mp3_files(folder):
  # get all files that are mp3 type
  onlyfiles = [os.path.abspath(path) for path in pathlib.Path(folder).rglob('*.mp3')]
  return onlyfiles

# Get all mp3 files recursively in folder
def get_all_wav_files(folder):
  # get all files that are mp3 type
  onlyfiles = [os.path.abspath(path) for path in pathlib.Path(folder).rglob('*.wav')]
  return onlyfiles
  
def transcribe_and_clasify(file):
  # Load your API key from an environment variable or secret management service
  keyfile = open("openai_api.key", "r")
  openai.api_key = keyfile.readline()
  keyfile.close()

  get_transcript(file)

  f = open(file.replace(".mp3", "_transcript.txt"), "r")
  input = f.read()
  f.close()

  chunks = text_to_chunks(input)

  chunk_summaries = []
  chunk_sentiments = []

  for chunk in chunks:
    chunk_summary = summarize_text(" ".join(chunk))
    #chunk_sentiment = classify_sentiment(" ".join(chunk))
    chunk_summaries.append(chunk_summary)
    #chunk_sentiments.append(chunk_sentiment)

  summary = " ".join(chunk_summaries)
  #sentiment = " ".join(chunk_sentiment)
  sentiment = classify_sentiment(summary)

  f = open(file.replace(".mp3", "result.txt"), "w")
  f.write("resumen: \n")
  f.write(summary + "\n")
  f.write("###########################################\n")
  f.write("sentimiento: \n")
  f.write(sentiment)
  f.close()
  #print(input)
  print(summary)
  print(sentiment)


#wavfiles = get_all_wav_files("./calls")
#print(wavfiles)
# resave all wav files using soundfile library to correct encoding for the pydub library
#resave_all_wav_files(wavfiles)
#sound = AudioSegment.from_wav("./calls/1140/test1.WAV")
#sound.export("./calls/1140/test1.mp3", format="mp3")
#convert_all_wav_to_mp3(wavfiles)
mp3files = get_all_mp3_files("./calls")
for f in mp3files:
  transcribe_and_clasify(f)
#print(mp3files)