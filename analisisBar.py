import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


nltk.download('stopwords')
nltk.download('wordnet')

mensajes_raw = []
with open('COMENTARIOS_MINERIA_DE_DATOS.txt', 'r', encoding='utf-8') as f:
    mensajes_raw = [linea.strip() for linea in f if linea.strip()]
mensajes = [msg.lstrip('*').strip() for msg in mensajes_raw]


def preprocesar_texto(texto):
 
    texto = texto.lower()
 
    texto = re.sub(r'\W', ' ', texto)
    texto = re.sub(r'\d', ' ', texto)

    palabras = texto.split()
 
    palabras = [palabra for palabra in palabras if palabra not in stopwords.words('spanish')]

    lematizador = WordNetLemmatizer()
    palabras = [lematizador.lemmatize(palabra) for palabra in palabras]
    return ' '.join(palabras)

def obtener_sentimiento(texto):
    analisis = TextBlob(texto)
    if analisis.sentiment.polarity > 0:
        return "Positivo"
    elif analisis.sentiment.polarity < 0:
        return "Negativo"
    else:
        return "Neutro"
mensajes_procesados = [preprocesar_texto(mensaje) for mensaje in mensajes]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mensajes_procesados)
mtd = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("Matriz de Términos-Documentos:")
print(mtd)

sentimientos = [obtener_sentimiento(mensaje) for mensaje in mensajes]

print("\nSentimiento de cada mensaje:")
for i in range(len(mensajes)):  
    print(f"{mensajes[i]} => {sentimientos[i]}")

conteo_sentimientos = Counter(sentimientos)

categorias = list(conteo_sentimientos.keys())
valores = list(conteo_sentimientos.values())

plt.figure(figsize=(8, 5))
sns.barplot(x=categorias, y=valores, palette="viridis")


plt.xlabel("Sentimiento")
plt.ylabel("Cantidad de Mensajes")
plt.title("Distribución de Sentimientos en los Mensajes")
plt.show()