import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required data
nltk.download('punkt')
nltk.download('wordnet')

# Multiple sentence input
print("Enter multiple sentences (Type END to finish):")

lines = []
while True:
    line = input()
    if line.upper() == "END":
        break
    lines.append(line)

text = " ".join(lines)

# Sentence Tokenization
sentences = sent_tokenize(text)

# Word Tokenization
tokens = word_tokenize(text)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# Display Results
print("\n===== ORIGINAL TEXT =====")
print(text)

print("\n===== SENTENCES =====")
for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")

print("\n===== TOKENS =====")
print(tokens)

print("\nTotal Sentences:", len(sentences))
print("Total Tokens:", len(tokens))

print("\n===== STEMMED WORDS =====")
print(stemmed_words)

print("\n===== LEMMATIZED WORDS =====")
print(lemmatized_words)

print("\n===== COMPARISON =====")
print("Original\t\tStemmed\t\tLemmatized")
print("-" * 50)

for original, stemmed, lemma in zip(tokens, stemmed_words, lemmatized_words):
    print(f"{original:15} {stemmed:15} {lemma}")

print("\n===== OBSERVATION =====")
print("1. Tokenization split the text into sentences and words.")
print("2. Stemming reduces words to root forms.")
print("3. Lemmatization converts words into meaningful dictionary forms.")
print("4. Lemmatization generally provides better preprocessing for NLP tasks.")