from vader_PT import SentimentIntensityAnalyzer

# Função para analisar o sentimento de uma sentença
def analyze_sentiment(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict['compound']

# Frase para análise
frase = "Para mim é uma das melhores francesinhas do Norte! Boas carnes, bom molho e excelente sabor! Atendimento rápido. Espaço acolhedor. Recomendo experimentar!"

# Analisar a polaridade da frase
polaridade = analyze_sentiment(frase)
print()
print("Frase:", frase)
print()
print("Polaridade:", polaridade)
print()

# Determinar se a polaridade é positiva, negativa ou neutra
if polaridade > 0.35:
    print("Sentimento: Positivo")
elif polaridade < -0.35:
    print("Sentimento: Negativo")
else:
    print("Sentimento: Neutro")
