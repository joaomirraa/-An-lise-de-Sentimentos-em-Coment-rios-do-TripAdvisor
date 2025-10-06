''' LeIA - Léxico para Inferência Adaptada
https://github.com/rafjaa/LeIA

Este projeto é um fork do léxico e ferramenta para análise de 
sentimentos VADER (Valence Aware Dictionary and sEntiment Reasoner) 
adaptado para textos em português.

Autor do VADER: C.J. Hutto
Repositório: https://github.com/cjhutto/vaderSentiment

'''

import re
import math
import unicodedata
from itertools import product
import os

PACKAGE_DIRECTORY = "C:\\Users\\Utilizador\\Desktop\\Projeto\\EntregaFinal_Projeto\\lexicons"

# Empirically derived mean sentiment intensity rating increase for booster words
B_INCR = 0.293
B_DECR = -0.293

# Empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word
C_INCR = 0.733
N_SCALAR = -0.74

# Remoção de Pontuação 
REGEX_REMOVE_PUNCTUATION = re.compile('[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))

PUNC_LIST = [
    ".", "!", "?", ",", ";", ":", "-", "'", "\"", "...",
    "—", "–", "!?", "?!", "!!", "!!!", "??", "???", "?!?", 
    "!?!", "?!?!", "!?!?"
]

# Negações
NEGATE = [t.strip() for t in open("C:\\Users\\Utilizador\\Desktop\\Projeto\\EntregaFinal_Projeto\\lexicons\\negate.txt")]

# Booster/dampener 'intensifiers' or 'degree adverbs' (Portuguese)
boosters = []
for boost in open("C:\\Users\\Utilizador\\Desktop\\Projeto\\EntregaFinal_Projeto\\lexicons\\booster.txt"):
    parts = boost.strip().split(' ')
    boosters.append([' '.join(parts[:-1]), parts[-1]])

BOOSTER_DICT = {}
for t, v in boosters: 
    BOOSTER_DICT[t] = B_INCR if v == 'INCR' else B_DECR


# Check for special case idioms containing lexicon words
SPECIAL_CASE_IDIOMS = {}


def negated(input_words, include_nt=True):
    """
    Determina se o input tem palavras negativas
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE) #mete na neg_words o que está NEGATE (extende)
    for word in neg_words:
        if word in input_words:
            return True
    # if include_nt:
    #     for word in input_words:
    #         if "n't" in word:
    #             return True
    return False

'''A função normalize tem como objetivo normalizar um valor de pontuação de sentimento para que esteja 
dentro de um intervalo específico, geralmente entre -1 e 1'''
def normalize(score, alpha=15): #o score é dado no léxico (parâmetro a seguir à palavra). varia de -4 a 4
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score

#Verifica se há palavvras com letra maiuscula conjuntamente com palavras em minusculas
def allcap_differential(words):
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1  #verifica quantas palavras estão em letra maiuscula
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True   #Retorna True se algumas palavras estiverem em maiúsculas e outras não estiverem   
    return is_different

'''verifica se palavras ou expressões anteriores a uma palavra específica em um texto aumentam, diminuem ou negam o sentimento associado a essa palavra
   ex: tão mau'''
def scalar_inc_dec(word, valence, is_cap_diff): #Recebe uma palavra, sentimento (valence) e uma flag que indica se outras palavras estão em maiúsculas (is_cap_diff)
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT: #Verifica se a palavra está presente no dicionário de intensificadores/atenuadores 
        scalar = BOOSTER_DICT[word_lower] #Obtém o valor do intensificador/atenuador associado à palavra no dicionário. INCR ou DECR
        if valence < 0: # Se o sentimento é negativo, inverte o sinal do intensificador/atenuador
            scalar *= -1
        
        # Check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff: #se a palavra estiver em maiusculas e houver uma diferença de capitalização no texto "ABSOLUTAMENTE incrível"
            if valence > 0:  #se o sentimento for pos
                scalar += C_INCR #entao incrementa-se o intensificador/atenuador. Ou seja, ter "extremamente bom" tem menor valor que "EXTREMAMENTE bom" 
            else:
                scalar -= C_INCR
    return scalar


class SentiText(object):
    """
    Identifica propriedades relevantes ao sentimento a nível de strings no texto de entrada.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    '''#constrói um dicionário que associa palavras e pontuações às suas formas combinadas'''
    def _words_plus_punc(self):
        """
        self.text = "Hello, world!"
        o resultado de _words_plus_punc pode ser algo assim:
        {
            ',Hello': 'Hello',
            '.Hello': 'Hello',
            '!Hello': 'Hello',
            '?Hello': 'Hello',
            ',world': 'world',
            '.world': 'world',
            '!world': 'world',
            '?world': 'world'
}
        """
        no_punc_text = REGEX_REMOVE_PUNCTUATION.sub('', self.text)  #Remove a pontuação do texto de entrada 
        
        #Divide o texto sem pontuação em uma lista de palavras
        words_only = no_punc_text.split()
        
        #Remove palavras únicas e mantém apenas aquelas que têm mais de uma letra
        words_only = set(w for w in words_only if len(w) > 1)
        
        #  cria um dicionário onde as chaves são strings formadas pela concatenação
        # de um caractere de pontuação com uma palavra, e os valores são o segundo elemento de cada par
        punc_before = {''.join(p): p[1] for p in product(PUNC_LIST, words_only)} #{',hello': 'hello', ',world': 'world', '.hello': 'hello'}
                                                                                 # product(p[0], p[1]) -> p[1] corresponde a words_only

        punc_after = {''.join(p): p[0] for p in product(words_only, PUNC_LIST)}  #{'hello,': 'hello', 'hello.': 'hello', 'hello!': 'hello'}
                                                                                 #product(p[0], p[1]) -> p[0] corresponde a words_only
                                                                                 
        #Combina os dicionários punc_before e punc_after, criando um dicionário final   
        words_punc_dict = punc_before
        words_punc_dict.update(punc_after)

        return words_punc_dict

    #está explicado na init
    def _words_and_emoticons(self):
        """
        Remove a pontuação no início e no final das palavras.
        Leaves contractions and most emoticons - contractions: you're, em portugues isto não há
        Mas não preserva emoticons formados por pontuação seguida de letra (por exemplo, :D)
        """
        wes = self.text.split()
        words_punc_dict = self._words_plus_punc()
        wes = [we for we in wes if len(we) > 1]  #remove possíveis espaços em branco ou caracteres soltos que não são palavras ou emoticons válidos
        for i, we in enumerate(wes):
            if we in words_punc_dict:
                wes[i] = words_punc_dict[we]
        return wes


class SentimentIntensityAnalyzer(object):
    def __init__(
            self,
            lexicon_file=os.path.join(
                PACKAGE_DIRECTORY,
                'vader_lexicon_ptbr.txt'
            ),
            emoji_lexicon=os.path.join(
                PACKAGE_DIRECTORY,
                'emoji_utf8_lexicon_ptbr.txt'
            )
    ):
        with open(lexicon_file, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        with open(emoji_lexicon, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()


    def make_lex_dict(self):
        """
        Converte o ficheiro do lexico para um dicionario 
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.split('\n'):
            if len(line) < 1:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict


    def make_emoji_dict(self):
        """
        Converte o lexico dos emogis para um dicionario
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.split('\n'):
            if len(line) < 1:
                continue
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict


    def polarity_scores(self, text):
        """
        Retorna um número decimal que representa a intensidade do sentimento com base no texto de entrada
        Valores positivos indicam uma valência positiva, enquanto valores negativos indicam uma valência negativa.
        """

        # Remove acentos
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

        ''' converte emojis para as suas descricoes textuais---> no lexico está "😀	rosto sorridente" '''
        text_token_list = text.split()
        text_no_emoji_lst = []
        for token in text_token_list:
            if token in self.emojis:
                # obtem a descrição do emoji
                description = self.emojis[token]
                text_no_emoji_lst.append(description)
            else:
                text_no_emoji_lst.append(token)
        text = " ".join(x for x in text_no_emoji_lst)

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for item in words_and_emoticons:
            valence = 0
            i = words_and_emoticons.index(item)
            # vê se há palavras no vader_lexicon que possam ser usadas como modificadores ou negações
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)
        valence_dict = self.score_valence(sentiments, text) #função score valence está definida mais a baixo

        return valence_dict


    def sentiment_valence(self, valence, sentitext, item, i, sentiments): 
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:

            # obtem a valence do sentimento
            valence = self.lexicon[item_lowercase]

            # Verifica se a palavra está toda em maiúsculas e se ha diferença entre maiúsculas e minúsculas
            if item.isupper() and is_cap_diff:
                # Ajusta a valência se a palavra estiver em maiúsculas
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # Dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon: # Verifica se a palavra anterior (ou emoji)
                                                                                                       #não está no lexico
                    #calcula o impacto do item anterior na valência do sentimento atual
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)  #scalar_inc_dec definida na linha 110 
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i) #verifica se há negações que afetam a valência do sentimento.
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_and_emoticons, i) #verifica expressões idiomáticas especiais que possam 
                                                                                              #afetar a valência do sentimento.

            # valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)

        return sentiments


    # TODO: Portuguese
    # def _least_check(self, valence, words_and_emoticons, i):
    #     # check for negation case using "least"
    #     if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon \
    #             and words_and_emoticons[i - 1].lower() == "least":
    #         if words_and_emoticons[i - 2].lower() != "at" and words_and_emoticons[i - 2].lower() != "very":
    #             valence = valence * N_SCALAR
    #     elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon \
    #             and words_and_emoticons[i - 1].lower() == "least":
    #         valence = valence * N_SCALAR
    #     return valence


    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # Verifica se há modificação na valência do sentimento devido à conjunção contrastante 'mas'.
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]

        for mas in ['mas', 'entretanto', 'todavia', 'porem', 'porém']:
            if mas in words_and_emoticons_lower:
                bi = words_and_emoticons_lower.index(mas)
                for sentiment in sentiments:
                    si = sentiments.index(sentiment)
                    if si < bi:
                        # Ajusta a valência para itens anteriores à conjunção 'mas'
                        sentiments.pop(si)
                        sentiments.insert(si, sentiment * 0.5)
                    elif si > bi:
                        # Ajusta a valência para itens posteriores à conjunção 'mas'
                        sentiments.pop(si)
                        sentiments.insert(si, sentiment * 1.5)
            return sentiments


    @staticmethod
    def _special_idioms_check(valence, words_and_emoticons, i):
        #Verifica e ajusta a valência do sentimento para expressões idiomáticas especiais.
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        onezero = "{0} {1}".format(
            words_and_emoticons_lower[i - 1], 
            words_and_emoticons_lower[i]
        )

        twoonezero = "{0} {1} {2}".format(
            words_and_emoticons_lower[i - 2],
            words_and_emoticons_lower[i - 1], 
            words_and_emoticons_lower[i]
        )

        twoone = "{0} {1}".format(
            words_and_emoticons_lower[i - 2], 
            words_and_emoticons_lower[i - 1]
        )

        threetwoone = "{0} {1} {2}".format(
            words_and_emoticons_lower[i - 3],
            words_and_emoticons_lower[i - 2], 
            words_and_emoticons_lower[i - 1]
        )

        threetwo = "{0} {1}".format(
            words_and_emoticons_lower[i - 3], 
            words_and_emoticons_lower[i - 2]
        )

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[seq]
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(
                words_and_emoticons_lower[i], 
                words_and_emoticons_lower[i + 1]
            )
            if zeroone in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroone]

        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(
                words_and_emoticons_lower[i], 
                words_and_emoticons_lower[i + 1],
                words_and_emoticons_lower[i + 2]
                )
            if zeroonetwo in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroonetwo]

        # Check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in BOOSTER_DICT:
                valence = valence + BOOSTER_DICT[n_gram]

        return valence


    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        '''Verifica e ajusta a valência do sentimento para expressões negativas
        '''
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 1- Verifica se a palavra é negada
                # Ajusta a valência para palavras negadas
                valence = valence * N_SCALAR
                
        #Verifica se as duas palavras anteriores estão a negar a valência
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == "nunca" and \
                    (words_and_emoticons_lower[i - 1] == "entao" or
                     words_and_emoticons_lower[i - 1] == "este"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 2] == "sem" and \
                    words_and_emoticons_lower[i - 1] == "dúvida":
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 2 words preceding the lexicon word position
                valence = valence * N_SCALAR
                
        # Verifica se as três palavras anteriores estão a negar a valência
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == "nunca" and \
                    (words_and_emoticons_lower[i - 2] == "entao" or words_and_emoticons_lower[i - 2] == "este") or \
                    (words_and_emoticons_lower[i - 1] == "entao" or words_and_emoticons_lower[i - 1] == "este"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 3] == "sem" and \
                    (words_and_emoticons_lower[i - 2] == "dúvida" or words_and_emoticons_lower[i - 1] == "dúvida"):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 3 words preceding the lexicon word position
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        '''Adiciona ênfase a partir de pontos de exclamação e interrogação.'''
        ep_amplifier = self._amplify_ep(text) # Amplifica devido a pontos de exclamação
        qm_amplifier = self._amplify_qm(text) # Amplifica devido a pontos de interrogação
        punct_emph_amplifier = ep_amplifier + qm_amplifier # Soma as ênfases

        return punct_emph_amplifier


    @staticmethod
    def _amplify_ep(text):
        '''Verifica a ênfase adicional resultante de pontos de exclamação (até 4 deles)'''
        # Conta a quantidade de pontos de exclamação no texto
        ep_count = text.count("!")
        # Limita a amplificação a até 4 pontos de exclamação
        if ep_count > 4:
            ep_count = 4
        # Aumento médio na valência devido a pontos de exclamação
        ep_amplifier = ep_count * 0.292

        return ep_amplifier


    @staticmethod
    def _amplify_qm(text):
        #Verifica a ênfase adicional resultante de pontos de interrogação (2 ou 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                 # Aumento médio na valência devido a pontos de interrogação
                qm_amplifier = qm_count * 0.18
                
            else:
                # Amplificação máxima devido a pontos de interrogação
                qm_amplifier = 0.96

        return qm_amplifier


    @staticmethod
    def _sift_sentiment_scores(sentiments):
        '''Separa os scores de sentimentos positivos e negativos.'''
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        # Calcula a soma dos scores positivos e negativos
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # Compensa palavras neutras contadas como 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  ## Compensa palavras neutras
            # Conta palavras neutras
            if sentiment_score == 0:
                neu_count += 1

        return pos_sum, neg_sum, neu_count


    def score_valence(self, sentiments, text):
        '''Calcula a valência do sentimento com base nos scores de sentimentos.'''
        if sentiments:
            sum_s = float(sum(sentiments)) # Soma os scores de sentimentos
            # Adiciona ênfase da pontuação no texto
            punct_emph_amplifier = self._punctuation_emphasis(text)
            # Ajusta a valência com base na ênfase da pontuação
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            # Normaliza o valor da valência (linha 78)
            compound = normalize(sum_s)
            # Separa os scores de sentimentos em positivos, negativos e neutros
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            # Adiciona a ênfase da pontuação aos scores de sentimentos positivos ou negativos
            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            # Calcula as porcentagens de sentimentos positivos, negativos e neutros
            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)
        
        # Caso não haja scores de sentimentos, retorna valores neutros
        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0
            
        # Retorna um dicionário com os valores de valência (arrendonda)
        sentiment_dict = {
            'neg': round(neg, 3),
            'neu': round(neu, 3),
            'pos': round(pos, 3),
            'compound': round(compound, 4)
        }

        return sentiment_dict


if __name__ == '__main__':
    pass

    # TODO: tests and examples (Portuguese)
