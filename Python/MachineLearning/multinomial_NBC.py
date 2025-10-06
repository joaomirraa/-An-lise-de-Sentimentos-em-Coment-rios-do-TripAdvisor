import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Passo 1: Ler os Dados
df = pd.read_excel("C:\\Users\\Utilizador\\Downloads\\comentarios_polaridade.xlsx")


# Passo 2: Extração de Características
vectorizer = CountVectorizer()  # Utilizar CountVectorizer para representação de saco de palavras
X = vectorizer.fit_transform(df['comentarios'])

# Converter pontuações de sentimento contínuas em etiquetas discretas
y = df['polaridade']

# Passo 3: Dividir os Dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Treinar o Classificador Naive Bayes Multinomial
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Passo 5: Avaliar o Classificador
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100  # Convertendo precisão para percentagem
print("Precisão do Naive Bayes:", accuracy)

# Gráfico da Distribuição de Polaridade nos Conjuntos de Treino e Teste
train_counts = y_train.value_counts()
test_counts = y_test.value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de pizza do conjunto de treino
axes[0].pie(train_counts, labels=train_counts.index, autopct='%1.1f%%', startangle=140)
axes[0].set_title('Distribuição de Polaridade no Conjunto de Treino')

# Gráfico de pizza do conjunto de teste
axes[1].pie(test_counts, labels=test_counts.index, autopct='%1.1f%%', startangle=140)
axes[1].set_title('Distribuição de Polaridade no Conjunto de Teste')

# Adicionar uma caixa de texto para a precisão
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
axes[1].text(0.05, 0.95, f'Precisão: {accuracy:.2f}%', transform=axes[1].transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

# Adicionar título ao gráfico
plt.suptitle('Naive Bayes Multinomial', fontsize=16)

plt.show()
