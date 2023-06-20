import os
from openai.embeddings_utils import cosine_similarity, get_embedding
import openai
import spacy
from spacy.lang.es import Spanish

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

    if chunk_total_words > 1000:
      chunks.append([])
      chunk_total_words = len(sentence.text.split(" "))

    chunks[len(chunks)-1].append(sentence.text)
  
  return chunks

#summarize text in no more than 3 sentences
def summarize_text(text):
  prompt = f"Summarize the following text in no more than 3 sentences:\n{text}"

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
  prompt = f"The follwing text is a summary of a phone conversation between a customer and an account manager. Classify the sentiment of the call into either positive, negative or neutral:\n{text}"

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
def get_transcript():
    audio_file= open("./testFile.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    file = open("transcript.txt", "w")
    file.write(transcript.text)
    file.close()

# Load your API key from an environment variable or secret management service
openai.api_key = "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"

#get_transcript()

f = open("transcript.txt", "r")
input = f.read()
f.close()

chunks = text_to_chunks(input)

chunk_summaries = []
chunk_sentiments = []

for chunk in chunks:
  chunk_summary = summarize_text(" ".join(chunk))
  chunk_sentiment = classify_sentiment(" ".join(chunk))
  chunk_summaries.append(chunk_summary)
  chunk_sentiments.append(chunk_sentiment)

summary = " ".join(chunk_summaries)
sentiment = " ".join(chunk_sentiment)
print(input)
print(summary)
print(sentiment)