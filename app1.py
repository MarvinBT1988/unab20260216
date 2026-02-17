import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import spacy
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


nltk.download('stopwords')
nlp = spacy.load("es_core_news_sm")


mensajes = [
    "En metro ayer nos atendieron bien y los desayunos son una delicia",
    "Los mejores desayunos... Y sin duda el mejor café, mi récord personal son 5 tazas en un desayuno!!!",
    "El servicio al cliente es excelente",
    "En el puerto de la libertad dan un pésimo servicio",
    "Pollo Campero El Salvador lo que más me encanta fue el pichel de café",
    "Por estar chambriando ni lo atienden a uno",
    "¡¡Eso no sirve ya!!  de café deberían mejorar, es horrible como le sirven agua caliente en TODOS los restaurantes",
    "Gran paja yo fui a metro al Campero y le dije al que me despachó '¿no hay hielo?' y me dijo 'si no sale es porque no hay', no es manera de hablarle a los clientes",
    "Pollo Campero El Salvador y los benedictinos???"
]

etiquetas = [1, 1, 1, 0, 1, 0, 0, 0, 1]


def preprocesar_texto(texto):
    texto = texto.lower()
 
    texto = re.sub(r'\W', ' ', texto)
    texto = re.sub(r'\d', ' ', texto)

    doc = nlp(texto)
    palabras = [token.lemma_ for token in doc if token.text not in stopwords.words('spanish')]
    return ' '.join(palabras)


mensajes_procesados = [preprocesar_texto(mensaje) for mensaje in mensajes]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(mensajes_procesados)

X_train, X_test, y_train, y_test = train_test_split(X, etiquetas, test_size=0.2, random_state=42)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)


predicciones = modelo.predict(X_test)
print("Precisión del modelo:", accuracy_score(y_test, predicciones))


texto_total = ' '.join(mensajes_procesados)
nube = WordCloud(width=800, height=400, background_color='white').generate(texto_total)

plt.figure(figsize=(10,5))
plt.imshow(nube, interpolation='bilinear')
plt.axis('off')
plt.show()