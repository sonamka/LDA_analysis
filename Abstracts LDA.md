```python
import sys 
import pandas as pd 
import numpy as np
import requests
import multiprocessing 
from multiprocessing import Pool
from itertools import repeat
import json
import sys
# !{sys.executable} -m spacy download en
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, logging, warnings
import gensim.corpora as corpora
# from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
# NLTK Stop words

# Remove stop words 
from gensim.utils import simple_preprocess
import nltk

nltk.download('stopwords')
# from nltk.corpus import stopwords
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]   Package stopwords is already up-to-date!

    True
    
```python
import spacy
print(spacy.__version__)
```
    3.8.5

```python
#Import data from excel
df = pd.read_excel('Abstracts.xlsx')
print (df.head())
```

                                                Abstract
    0  Designing and deploying artificial intelligenc...
    1  Agriculture faces increasing challenges all ar...
    2  Sustained intensification in agricultural prod...
    3  With support from the Centers of Research Exce...
    4  This award is funded in whole or in part under...
    


```python
# Merge all columns into a single series
merged_series = df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Convert the series to a single string
combined_text = ' '.join(merged_series)

print(combined_text)
```

    Designing and deploying artificial intelligence (AI) tools in agriculture represents an exciting opportunity for international collaboration, uniting diverse expertise and resources to tackle global challenges. Our project aims to impact agriculture by developing, deploying, and democratizing AI tools to help farmers manage pests and stressors more effectively, making farming less risky, more profitable, and more sustainable. We plan to create AI-driven tools that provide personalized management advice, enhance crop yields, and support sustainable farming practices. This initiative will bring together scientists and practitioners from the US, India, and Japan, fostering international collaboration and innovation. The AI-driven approaches will benefit small and medium-sized farmers, offering easy-to-use, accessible technology to help them pursue climate-smart agriculture. The project also includes educational components and multilateral engagements to inspire the next generation of agricultural and AI experts. This EAGER project seeks to pursue multilateral research partnerships between the US, India, and Japan to develop and deploy AI-driven tools to enhance agricultural productivity. This team will work across two areas of collaborative effort: (i) developing hybrid machine learning models that combine sensor (proximal and remote) data with biophysical knowledge for yield and stress prediction, and (ii) utilizing agronomic data -- both biotic (insects, weeds, diseases) and abiotic (nutrient deficiencies, herbicide injury) -- to fine-tune and deploy large vision and language models developed by AIIRA ...
    


```python
#OPTIONAL: convert to string if you want to clean or manipulate any further
data = str(combined_text)
print(data)
```

    Designing and deploying artificial intelligence (AI) tools in agriculture represents an exciting opportunity for international collaboration, uniting diverse expertise and resources to tackle global challenges. Our project aims to impact agriculture by developing, deploying, and democratizing AI tools to help farmers manage pests and stressors more effectively, making farming less risky, more profitable, and more sustainable. We plan to create AI-driven tools that provide personalized management advice, enhance crop yields, and support sustainable farming practices. This initiative will bring together scientists and practitioners from the US, India, and Japan, fostering international collaboration and innovation. The AI-driven approaches will benefit small and medium-sized farmers, offering easy-to-use, accessible technology to help them pursue climate-smart agriculture. The project also includes educational components and multilateral engagements to inspire the next generation of agricultural and AI experts. This EAGER project seeks to pursue multilateral research partnerships between the US, India, and Japan to develop and deploy AI-driven tools to enhance agricultural productivity. This team will work across two areas of collaborative effort: (i) developing hybrid machine learning models that combine sensor (proximal and remote) data with biophysical knowledge for yield and stress prediction, and (ii) utilizing agronomic data -- both biotic (insects, weeds, diseases) and abiotic (nutrient deficiencies, herbicide injury) -- to fine-tune and deploy large vision and language models developed by AIIRA ...
    


```python
#This cell sets up our stop words list for terms that do not have much meaning in text analysis. The ipynb file has a shortened version of the stopwords_add to allow additional words depending on the text you're using.
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

stopwords_add= ['from', 'subject', 're', 'edu', 'use','also', 'well', 'including', 'simple', 'yet',  'work', 'inform',  'generally','given', 'work', 'help', 'play', 
                'et', 'al','across', 'towards','allow', 'body','affect','give', 'addtion','know', 'whose','real','open','often','although','act','find','even',
              'suggest','connect','tightly','release','space','function','regulate','control','role','apt','process', 'finely', 'tune','regulation','critical','functioning',
              'endeavor','finely','conditions','term','project','using','changes','fundamental', 'build','block','make','addition', 'provide', 'train','individual',
              'feasible', 'complex','mechanisms', 'remain', 'unclear', 'overarch','system', 'thereby', 'movement', 'valuable', 'educational','consider','waste',
              'study', 'field', 'establish','award','understand','expression','community','reach', 'the', 'on', 'in',
              'statutory','statutory','head','call','group','new','size','behavior','leave','internalize',
              'via','ecosystem','model','organisms','harsh','concurrently','precise','small','essential','long','release',
               'random','live','range','really','form','global','potential','collection','important','exposure','extreame','order', 'recipient','extreme','break','still','together',
              'cells','build','block', 'live','however','dont', 'exactly', 'cells', 'form', 'first', 'place', 'dont','molecules','cell', 'finally','question', 'challenge', 'pose', 'require',
               'significant','edit', 'hold','regulator','put','cant','decline', 'interact','attendance','major', 'systems','alter','host','many','may','must','support',
              'think','time','defend','wonder','adjustable','operations', 'meet','lack','conduct','whether','trait','directly','effect'
               ,'enhance','consumer', 'demand', 'value','address', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
                'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
                'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
                'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                 "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "pattern", "practices", 
               'lesson', 'instructor', "nsf", "represent", "technolog", "collaboration", "artificial", "intelligence", 
               "agricultural", "agriculture", "ai"]


stop_words.update(stopwords_add)
# stopwords.extend(stopwords_add)

%matplotlib inline
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
#This function ensures that the text is a string, removes punctuation and converts the text to lower case. 
#Then the text is split and stop words are filtered out. 
#You can also tokenize the text in this step if preferred.
def remove_stopwords(combined_text):
    import string
    # nltk.download('punkt')  # Download tokenizer if not already done
    text = combined_text.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = text.split()
    # tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)
filtered_text = remove_stopwords(combined_text)
print(filtered_text)
```

    designing deploying tools represents exciting opportunity international uniting diverse expertise resources tackle challenges aims impact developing deploying democratizing tools farmers manage pests stressors effectively making farming less risky profitable sustainable plan create aidriven tools personalized management advice crop yields sustainable farming initiative bring scientists practitioners us india japan fostering international innovation aidriven approaches benefit mediumsized farmers offering easytouse accessible technology pursue climatesmart includes components multilateral engagements inspire next generation experts eager seeks pursue multilateral research partnerships us india japan develop deploy aidriven tools productivity team two areas collaborative effort developing hybrid machine learning models combine sensor proximal remote data biophysical knowledge yield stress prediction ii utilizing agronomic data biotic insects weeds diseases abiotic nutrient deficiencies herbicide injury finetune deploy large vision language models developed aiira one five nifafunded national institutes us led iowa state university isu collaborating international partners span diverse environments aim develop validate robust scalable framework management supports realtime decisionmaking fosters sustainable globally initiative emphasizes outreach promoting interdisciplinary learning broadening participation aidriven funded part quad aiengage initiative national science foundation commonwealth scientific industrial research organization australia indian council research japan science technology agency advance innovation empower next generation reflects nsfs mission deemed worthy evaluation foundations intellectual merit broader impacts review criteria faces increasing challenges around world face climatechange related issues focus climate smart catalyzed developments machine learning needed better manage cyberphysical maintain productivity ensure food security constraining impact greenhouse ...
    


```python
from nltk.tokenize import WhitespaceTokenizer

# Tokenize here if you did not tokenize in the previous function remove_stopwords. 
#The WhitespaceTokenizer just splits the text based on spaces, while word tokenizer splits text by each word.
#Either one is fine to use based on how specific you want to be to split the text.
tokenizer_w = WhitespaceTokenizer()
tokenized_text = tokenizer_w.tokenize(filtered_text)
print(tokenized_text)
```

    ['designing', 'deploying', 'tools', 'represents', 'exciting', 'opportunity', 'international', 'uniting', 'diverse', 'expertise', 'resources', 'tackle', 'challenges', 'aims', 'impact', 'developing', 'deploying', 'democratizing', 'tools', 'farmers', 'manage', 'pests', 'stressors', 'effectively', 'making', 'farming', 'less', 'risky', 'profitable', 'sustainable', 'plan', 'create', 'aidriven', 'tools', 'personalized', 'management', 'advice', 'crop', 'yields', 'sustainable', 'farming', 'initiative', 'bring', 'scientists', 'practitioners', 'us', 'india', 'japan', 'fostering', 'international', 'innovation', 'aidriven', 'approaches', 'benefit', 'mediumsized', 'farmers', 'offering', 'easytouse', 'accessible', 'technology', 'pursue', 'climatesmart', 'includes', 'components', 'multilateral', 'engagements', 'inspire', 'next', 'generation', 'experts', 'eager', 'seeks', 'pursue', 'multilateral', 'research', 'partnerships', 'us', 'india', 'japan', 'develop', 'deploy', 'aidriven', 'tools', 'productivity', 'team', 'two', 'areas', 'collaborative', 'effort', 'developing', 'hybrid', 'machine', 'learning', 'models', 'combine', 'sensor', 'proximal', 'remote', 'data', 'biophysical', 'knowledge', 'yield', 'stress', 'prediction', 'ii', 'utilizing', 'agronomic', 'data', 'biotic', 'insects', 'weeds', 'diseases', 'abiotic', 'nutrient', 'deficiencies', 'herbicide', 'injury', 'finetune', 'deploy', 'large', 'vision', 'language', 'models', 'developed', 'aiira', 'one', 'five', 'nifafunded', 'national', 'institutes', 'us', 'led', 'iowa', 'state', 'university', 'isu', 'collaborating', 'international', 'partners', 'span', 'diverse', 'environments', 'aim', 'develop', 'validate', 'robust', 'scalable', 'framework', 'management', 'supports', 'realtime', 'decisionmaking', 'fosters', 'sustainable', 'globally', 'initiative', 'emphasizes', 'outreach', 'promoting', 'interdisciplinary', 'learning', 'broadening', 'participation', 'aidriven', 'funded', 'part', 'quad', 'aiengage', 'initiative', 'national', 'science', 'foundation', 'commonwealth', 'scientific', 'industrial', 'research', 'organization', 'australia', 'indian', 'council', 'research', 'japan', 'science', 'technology', 'agency', 'advance', 'innovation', 'empower', 'next', 'generation', 'reflects', 'nsfs', 'mission', 'deemed', 'worthy', 'evaluation', 'foundations', 'intellectual', 'merit', 'broader', 'impacts', 'review', 'criteria', 'faces', 'increasing', 'challenges', 'around', 'world', 'face', 'climatechange', 'related', 'issues', 'focus', 'climate', 'smart', 'catalyzed', 'developments', 'machine', 'learning', 'needed', 'better', 'manage', 'cyberphysical', 'maintain', 'productivity', 'ensure',... ]
    


```python
#Tokenizing text turns the text into a list
type(tokenized_text)
```

    list


```python
#I do not prefer distracting quotes in the tokenized text, so this cell removes the extra punctuation.
token_clean_text = [re.sub("\'", "", str(tokenized_text)) for sent in tokenized_text]

print(token_clean_text[:1])
```

    ['[designing, deploying, tools, represents, exciting, opportunity, international, uniting, diverse, expertise, resources, tackle, challenges, aims, impact, developing, deploying, democratizing, tools, farmers, manage, pests, stressors, effectively, making, farming, less, risky, profitable, sustainable, plan, create, aidriven, tools, personalized, management, advice, crop, yields, sustainable, farming, initiative, bring, scientists, practitioners, us, india, japan, fostering, international, innovation, aidriven, approaches, benefit, mediumsized, farmers, offering, easytouse, accessible, technology, pursue, climatesmart, includes, components, multilateral, engagements, inspire, next, generation, experts, eager, seeks, pursue, multilateral, research, partnerships, us, india, japan, develop, deploy, aidriven, tools, productivity, team, two, areas, collaborative, effort, developing, hybrid, machine, learning, models, combine, sensor, proximal, remote, data, biophysical, knowledge, yield, stress, prediction, ii, utilizing, agronomic, data, biotic, insects, weeds, diseases, abiotic, nutrient, deficiencies, herbicide, injury, finetune, deploy, large, vision, language, models, developed, aiira, one, five, nifafunded, national, institutes, us, led, iowa, state, university, isu, collaborating, international, partners, span, diverse, environments, aim, develop, validate, robust, scalable, framework, management, supports, realtime, decisionmaking, fosters, sustainable, globally, initiative, emphasizes, outreach, promoting, interdisciplinary, learning, broadening, participation, aidriven, funded, part, quad, aiengage, initiative, national, science, foundation, commonwealth, scientific, industrial, research, organization, australia, indian, council, research, japan, science, technology, agency, advance, innovation, empower, next, generation, reflects, nsfs, mission, deemed, worthy, evaluation, foundations, intellectual, merit, broader, impacts, review, criteria, faces, increasing, challenges, around, world, face, climatechange, related, issues, focus, climate, smart, catalyzed, developments, machine, learning, needed, better, manage, cyberphysical, maintain, productivity, ensure, food, security, constraining, impact, greenhouse, gas, emissions, proposal, us, participation, workshop, internet, things, iotartificial, intelligenceaimachine, learning, mlenabled, precision, hosted, bengurion, university, beersheva, july, 2023, focus, big, data, acquisition, aimlbased, analytics, precision, cover, wide, topics, focused, scoping, key, challenges, remaining, iotaiml, development, sustainable, reflects, nsfs, mission, deemed, worthy, evaluation, foundations, intellectual, merit, broader, impacts, review, criteria, sustained, intensification, production, caloric, needs, rapidly, increasing, population, combating, impacts, climate, change, â€“, two, grand, challenges, 21st, century, creates, threeway, agaid, institute, one, five, nifafunded, institutes, us, led, washington, state,...]
    


```python
#Since tokenizing the text turned it into a list as we checked earlier, we are converting back to a string.
token_clean_strg = str(token_clean_text)
```


```python
#This step is not necessary but I am showing it here because this is how I checked how large my data was after processing.
len(token_clean_text)
```
    6520

```python
#Now we still need to stem the words which means that the words will be deduced to their base terms. 
#Stemming preprocesses the text for LDA analysis.
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
nltk.download ('punkt_tab')

stemmer = SnowballStemmer("english", True)
words = word_tokenize(token_clean_strg)
stemmed_words = [stemmer.stem(word) for word in words]
```

    [nltk_data] Downloading package punkt_tab to
    [nltk_data]   Package punkt_tab is already up-to-date!
    


```python
type(stemmed_words)
```

    list

```python
stemmed_words_strg = str(stemmed_words)
```


```python
#This step is not necessary, again, just checking.
len(stemmed_words_strg)
```

    651556645


```python
#We are merely cleaning the text a little more and the next 2 cells are just to view the text, optional
cleaned_texts = [sent.replace("'", "").replace(",", "").replace(":", "").replace(".", "").replace("(", "").replace(")", "") for sent in stemmed_words_strg]
```


```python
with open('output.txt', 'w', encoding='utf-8') as f:
     for item in cleaned_texts:         
         f.write(item)
print("File saved successfully as 'output.txt'")
```

    File saved successfully as 'output.txt'
    


```python
#optional if you want to see the cleaned text
file_obj = open("output.txt", "r", encoding="utf-8")
print(file_obj.read(1000))
```

    [[ "" [ design  deploy  tool  repres  excit  opportun  intern  unit  divers  expertis  resourc  tackl  challeng  aim  impact  develop  deploy  democrat  tool  farmer  manag  pest  stressor  effect  make  farm  less  riski  profit  sustain  plan  creat  aidriven  tool  person  manag  advic  crop  yield  sustain  farm  initi  bring  scientist  practition  us  india  japan  foster  intern  innov  aidriven  approach  benefit  mediums  farmer  offer  easytous  access  technolog  pursu  climatesmart  includ  compon  multilater  engag  inspir  next  generat  expert  eager  seek  pursu  multilater  research  partnership  us  india  japan  develop  deploy  aidriven  tool  product  team  two  area  collabor  effort  develop  hybrid  machin  learn  model  combin  sensor  proxim  remot  data  biophys  knowledg  yield  stress  predict  ii  util  agronom  data  biotic  insect  weed  diseas  abiot  nutrient  defici  herbicid  injuri  finetun  deploy  larg  vision  languag  model  develop  aiira  one 
    


```python
cleaned_strg1 = ''.join(map(str,cleaned_texts))
```


```python
len(cleaned_strg1)
```

    353781724


```python
#This step is lemmatizing the text which standardizes words by reducing to base form for better analysis.
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:         
        return wordnet.NOUN
       
def lemmatize_passage(cleaned_strg1):
    words = word_tokenize(cleaned_strg1)
    pos_tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence

result = lemmatize_passage(cleaned_strg1)
```

    [nltk_data] Downloading package averaged_perceptron_tagger_eng to
    [nltk_data]   Package averaged_perceptron_tagger_eng is already up-to-
    [nltk_data]       date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]   Package wordnet is already up-to-date!
    


```python
#It's just good to know what I'm working with. This is optional.
type(result)
```

    str


```python
with open('result.txt', 'w', encoding='utf-8') as f:
     for item in result:         
         f.write(item)
print("File saved successfully as 'result.txt'")
```

    File saved successfully as 'result.txt'
    


```python
#The next step is vectorizing the text and we need to change the string to a list.
def Convert(string):
    li = list(string.split(" "))
    return li
list_result = Convert(result)
```


```python
#Again, optional check.
type(list_result)
```

    list


```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
#LDA topic model algorithm requires a document word matrix, which is what we are creating here.
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             #stop_words='english',             # remove stop words
                             #lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}'  # num chars > 3
                             # max_features=50000,             # max number of unique words
                            )

data_vectorized = vectorizer.fit_transform(list_result)
```


```python
print(data_vectorized[:10])
```

      (5, 351)	1
      (6, 348)	1
      (7, 1290)	1
      (8, 1078)	1
      (9, 475)	1
    


```python
vector = data_vectorized
```


```python
#Optional check to see how large your vector is.
vector.shape
```

    (42595164, 1416)


```python
#This code selects random rows fom the vector because the size of the vector may be too large 
#for available computing capacity.
import numpy as np
from scipy.sparse import csr_matrix

def select_random_rows(matrix, num_rows):
    # Get the number of rows in the matrix
    total_rows = matrix.shape[0]
    
    # Generate random row indices
    random_indices = np.random.choice(total_rows, size=num_rows, replace=False)
    
    # Extract the selected rows
    selected_rows = matrix[random_indices]
    
    # Create a new csr_matrix from the selected rows
    new_matrix = csr_matrix(selected_rows)
    
    return new_matrix

num_rows_to_select = 1000000
new_matrix = select_random_rows(vector, num_rows_to_select)

print(new_matrix[:10])
```

      (0, 88)	1
      (1, 957)	1
      (2, 417)	1
      (3, 1089)	1
      (5, 668)	1
      (6, 930)	1
      (7, 1340)	1
      (8, 1257)	1
      (9, 979)	1
    


```python
#Optional, just checking again. I needed a smaller vector, and the randomization will be easier to work with on my machine.
new_matrix.shape
```

    (1000000, 1416)


```python
# Sparsicity is the percentage of non-zero datapoints in the document-word matrix, that is data_vectorized.
# Since most cells in this matrix will be zero, I am interested in knowing what percentage of cells --
# contain non-zero values.
data_dense = new_matrix.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
```


```python
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools if you decide to create visuals
#import pyLDAvis
#import pyLDAvis.sklearn
#import matplotlib.pyplot as plt
#%matplotlib inline
import spacy
#import en_core_web_sm
```


```python
# Build LDA Model

lda_model = LatentDirichletAllocation(n_components=10,            # Number of topics
                                      max_iter=5,                # Max learning iterations
                                      learning_method='online',  # The method used for learning 'online' indicates online variational Bayes
                                      #random_state=100,          # Random state When you set a specific value for random_state, you guarantee that the same data points will be included in the training and testing sets every time you run the code.
                                      batch_size=200,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: -1 skips the perplexity
                                      n_jobs = -1,               # Use all available CPUs; -1 uses all available cores
                                     )
lda_output = lda_model.fit_transform(new_matrix)

print(lda_model)
```

    LatentDirichletAllocation(batch_size=200, learning_method='online', max_iter=5,
                              n_jobs=-1)
    


```python
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(new_matrix))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(new_matrix))

# See model parameters
pprint(lda_model.get_params())
```

    Log Likelihood:  -6392977.39616302
    Perplexity:  693.8334000611763
    {'batch_size': 200,
     'doc_topic_prior': None,
     'evaluate_every': -1,
     'learning_decay': 0.7,
     'learning_method': 'online',
     'learning_offset': 10.0,
     'max_doc_update_iter': 100,
     'max_iter': 5,
     'mean_change_tol': 0.001,
     'n_components': 10,
     'n_jobs': -1,
     'perp_tol': 0.1,
     'random_state': None,
     'topic_word_prior': None,
     'total_samples': 1000000.0,
     'verbose': 0}
    


```python
#Here we are creating search parameters to see which ones work best for the model
search_params = {'n_components': [5, 8, 12, 16], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(new_matrix)
```

             param_grid={&#x27;learning_decay&#x27;: [0.5, 0.7, 0.9],
                         &#x27;n_components&#x27;: [5, 8, 12, 16]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. 
             param_grid={&#x27;learning_decay&#x27;: [0.5, 0.7, 0.9],
                         &#x27;n_components&#x27;: [5, 8, 12, 16]})




```python
#Can skip this or use depending on if you want to use the method above.
GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=100, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=7, n_jobs=1,
             perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       n_jobs=1,
       param_grid={'n_topics': [5, 8, 12, 16], 'learning_decay': [0.5, 0.7, 0.9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
```




    GridSearchCV(error_score='raise',
                 estimator=LatentDirichletAllocation(batch_size=100,
                                                     learning_method=None,
                                                     n_components=7, n_jobs=1),
                 n_jobs=1,
                 param_grid={'learning_decay': [0.5, 0.7, 0.9],
                             'n_topics': [5, 8, 12, 16]},
                 return_train_score='warn')




```python
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(new_matrix))

```

    Best Model's Params:  {'learning_decay': 0.9, 'n_components': 5}
    Best Log Likelihood Score:  -1288081.7278126136
    Model Perplexity:  693.2089881828424
    


```python
# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)
topicnames = [f'Topic {i}' for i in range(best_lda_model.n_components)]
# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names_out()
df_topic_keywords.index = topicnames

# View
df_topic_keywords
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>105</th>
      <th>1115</th>
      <th>1172</th>
      <th>2009</th>
      <th>2021</th>
      <th>2023</th>
      <th>2025</th>
      <th>2050</th>
      <th>21st</th>
      <th>5axi</th>
      <th>...</th>
      <th>world</th>
      <th>worldwid</th>
      <th>worthi</th>
      <th>wsu</th>
      <th>wvsu</th>
      <th>wvu</th>
      <th>wyom</th>
      <th>x9d</th>
      <th>year</th>
      <th>yield</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Topic 0</th>
      <td>0.200034</td>
      <td>142.199871</td>
      <td>174.199871</td>
      <td>0.200038</td>
      <td>0.200034</td>
      <td>0.200038</td>
      <td>0.200034</td>
      <td>0.200033</td>
      <td>0.200034</td>
      <td>0.200039</td>
      <td>...</td>
      <td>0.200033</td>
      <td>0.200038</td>
      <td>0.200034</td>
      <td>0.200033</td>
      <td>1237.199871</td>
      <td>0.200038</td>
      <td>0.200039</td>
      <td>0.200018</td>
      <td>464.199871</td>
      <td>0.200034</td>
    </tr>
    <tr>
      <th>Topic 1</th>
      <td>173.199872</td>
      <td>0.200034</td>
      <td>0.200034</td>
      <td>0.200039</td>
      <td>0.200034</td>
      <td>0.200039</td>
      <td>623.199872</td>
      <td>0.200034</td>
      <td>0.200034</td>
      <td>0.200039</td>
      <td>...</td>
      <td>0.200034</td>
      <td>0.200039</td>
      <td>4372.199872</td>
      <td>0.200034</td>
      <td>0.200034</td>
      <td>0.200039</td>
      <td>0.200039</td>
      <td>0.200018</td>
      <td>0.200034</td>
      <td>1411.199872</td>
    </tr>
    <tr>
      <th>Topic 2</th>
      <td>0.200029</td>
      <td>0.200030</td>
      <td>0.200030</td>
      <td>0.200033</td>
      <td>0.200029</td>
      <td>0.200033</td>
      <td>0.200029</td>
      <td>0.200029</td>
      <td>0.200029</td>
      <td>299.199847</td>
      <td>...</td>
      <td>0.200029</td>
      <td>0.200033</td>
      <td>0.200029</td>
      <td>0.200029</td>
      <td>0.200030</td>
      <td>0.200033</td>
      <td>294.199847</td>
      <td>0.200855</td>
      <td>0.200030</td>
      <td>0.200029</td>
    </tr>
    <tr>
      <th>Topic 3</th>
      <td>0.200030</td>
      <td>0.200030</td>
      <td>0.200030</td>
      <td>142.199851</td>
      <td>0.200030</td>
      <td>349.199851</td>
      <td>0.200030</td>
      <td>0.200030</td>
      <td>0.200030</td>
      <td>0.200035</td>
      <td>...</td>
      <td>0.200030</td>
      <td>160.199851</td>
      <td>0.200030</td>
      <td>0.200030</td>
      <td>0.200030</td>
      <td>295.199851</td>
      <td>0.200035</td>
      <td>0.200016</td>
      <td>0.200030</td>
      <td>0.200030</td>
    </tr>
    <tr>
      <th>Topic 4</th>
      <td>0.200034</td>
      <td>0.200035</td>
      <td>0.200035</td>
      <td>0.200039</td>
      <td>144.199874</td>
      <td>0.200039</td>
      <td>0.200034</td>
      <td>316.199874</td>
      <td>143.199874</td>
      <td>0.200040</td>
      <td>...</td>
      <td>561.199874</td>
      <td>0.200039</td>
      <td>0.200034</td>
      <td>156.199874</td>
      <td>0.200035</td>
      <td>0.200039</td>
      <td>0.200040</td>
      <td>811.199092</td>
      <td>0.200035</td>
      <td>0.200034</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1416 columns</p>
</div>




```python
df_topic_keywords.to_csv('LDAtopics.csv')
```


```python
# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Word 0</th>
      <th>Word 1</th>
      <th>Word 2</th>
      <th>Word 3</th>
      <th>Word 4</th>
      <th>Word 5</th>
      <th>Word 6</th>
      <th>Word 7</th>
      <th>Word 8</th>
      <th>Word 9</th>
      <th>Word 10</th>
      <th>Word 11</th>
      <th>Word 12</th>
      <th>Word 13</th>
      <th>Word 14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Topic 0</th>
      <td>develop</td>
      <td>student</td>
      <td>institut</td>
      <td>food</td>
      <td>applic</td>
      <td>plant</td>
      <td>review</td>
      <td>includ</td>
      <td>method</td>
      <td>graduat</td>
      <td>region</td>
      <td>precis</td>
      <td>interdisciplinari</td>
      <td>plan</td>
      <td>particip</td>
    </tr>
    <tr>
      <th>Topic 1</th>
      <td>learn</td>
      <td>advanc</td>
      <td>foundat</td>
      <td>intellectu</td>
      <td>worthi</td>
      <td>design</td>
      <td>educ</td>
      <td>comput</td>
      <td>energi</td>
      <td>rural</td>
      <td>area</td>
      <td>manufactur</td>
      <td>improv</td>
      <td>structur</td>
      <td>explor</td>
    </tr>
    <tr>
      <th>Topic 2</th>
      <td>univers</td>
      <td>impact</td>
      <td>challeng</td>
      <td>broad</td>
      <td>sustain</td>
      <td>train</td>
      <td>innov</td>
      <td>model</td>
      <td>mission</td>
      <td>deem</td>
      <td>nsfs</td>
      <td>domain</td>
      <td>collabor</td>
      <td>state</td>
      <td>involv</td>
    </tr>
    <tr>
      <th>Topic 3</th>
      <td>research</td>
      <td>data</td>
      <td>scienc</td>
      <td>evalu</td>
      <td>program</td>
      <td>industri</td>
      <td>reflect</td>
      <td>merit</td>
      <td>need</td>
      <td>criterion</td>
      <td>workforc</td>
      <td>farm</td>
      <td>machin</td>
      <td>network</td>
      <td>isac</td>
    </tr>
    <tr>
      <th>Topic 4</th>
      <td>technolog</td>
      <td>crop</td>
      <td>infrastructur</td>
      <td>integr</td>
      <td>solut</td>
      <td>approach</td>
      <td>problem</td>
      <td>capabl</td>
      <td>enabl</td>
      <td>propos</td>
      <td>aim</td>
      <td>partnership</td>
      <td>associ</td>
      <td>land</td>
      <td>manag</td>
    </tr>
  </tbody>
</table>
</div>




```python
import joblib

# Save the necessary components
joblib.dump(best_lda_model, 'lda_model.pkl')  # Save LDA model
joblib.dump(new_matrix, 'new_matrix.pkl')    # Save document-term matrix
joblib.dump(vectorizer, 'vectorizer.pkl')    # Save vectorizer
```


    ['vectorizer.pkl']


```python

```
