import numpy as np
import pandas as pd

pd.set_option('display.max_columns',25) # Display number of columns in csv file.
pd.set_option('display.width',120) # set width of displaying columns.

movie = pd.read_csv('C:\\Users\\admin\\Machine Learning Algo\\mini pro\\Recommender_System\\tmdb_5000_movies.csv')
credit = pd.read_csv('C:\\Users\\admin\\Machine Learning Algo\\mini pro\\Recommender_System\\tmdb_5000_credits.csv')

print("*********************************************** Columns of Movie.csv file ***********************************************")
print(movie.columns)

print("\n*********************************************** Read Movie.csv file ***********************************************")
# print(movie.head(5))


print("\n*********************************************** Read credits.csv file ***********************************************")
# credit.head(5)
# credit.head(1)['crew'].values


##########################################################################
#           Merge movie and ceredits dataset
#########################################################################
movies = movie.merge(credit, on='title')
movies.head(1)


##########################################################################
#          Drop unnesesary columns in our moveis dataset
#########################################################################

# mit after chack with original_language column becouse in most case english language
drop_columns = ['budget', 'homepage', 'original_language', 'original_title', 'popularity', 
                'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime',
                'spoken_languages', 'status', 'tagline', 'vote_average', 'vote_count', 'id']

# movies = movies.drop(drop_columns, axis=1)
# movies
movies = movies[['movie_id','title','overview', 'genres', 'keywords', 'cast', 'crew']]
# movies


##########################################################################
#           finde missing data and Dublicate data
#########################################################################
movies.isnull().sum()


## Remove Missing Value Row
movies.dropna(inplace=True)
movies.isnull().sum()


## see dublicate data
movies.duplicated().sum()


# movies.iloc[0].genres


## In over dataset some column have innessesary data like id
## For example, in genres have two key id and name and we need only name so we can preprocess that column  
## in column data in str formate we can convert in to list
## use ast module for converting str to list

import ast
def convert(object):
    L =[]
    for i in ast.literal_eval(object): # using this opresen convert str to list
        L.append(i['name'])
    return L

# apply in genres
movies['genres'] = movies['genres'].apply(convert)

# movies.head()

# apply in keywords
movies['keywords'] = movies['keywords'].apply(convert)


def convert2(object):
    L =[]
    counter = 0
    for i in ast.literal_eval(object): # using this opresen convert str to list
        if counter != 5:
            L.append(i['name'])
            counter +=1
        else:
            break
    return L

# apply in cast
movies['cast'] = movies['cast'].apply(convert2)
# movies
# movies


# Crew in we calecte data where job == Director his name (list out Directer Name)

def Fetch_directer(object):
    L =[]
    for i in ast.literal_eval(object): # using this opresen convert str to list
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# apply in crew
movies['crew'] = movies['crew'].apply(Fetch_directer)
# movies


## overview: convert in to list using spliting
movies['overview'] = movies['overview'].apply(lambda x:x.split())
# movies.head()

## we can apply transfortation : remove white space between two words
## For example: [Sam Worthington] == [SamWorthington] like this

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()


## Now create tag
# in tag we can concatinet : overview + genres + keywords + cast + crew
## create tag column

movies['tag'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
# movies


# now we create new dataframe (df)

df = movies[['movie_id', 'title', 'tag']]
df['tag'] = df['tag'].apply(lambda x:" ".join(x)) # remove ,(coma)
# df.head()

# tag : all words convert in to small charechter
df['tag'] = df['tag'].apply(lambda x:x.lower())
df.head()


## Apply stamming using nltk library
## in nltk.stem.porter import PorterStemmer :: remove similer words and convert in to one word
## For Example : ['loving', 'loved', 'love'] convrt in to ['love', 'love', 'love']

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# ps.stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver stephenlang michellerodriguez jamescameron')
df['tag'] = df['tag'].apply(stem)
# print(df)

## tag(txt) convert to Vactor : textvactorizetion (we use bag of words technics)
## In Bag of words : step 1:: we can combain all tags (larg text)
##                   step 2:: Find most common words into larg text (use max_features attribut = No.)
##                   step 3:: Create tabel in store numbers of words by perticuler tags (which find in step 2)
##                             and Tabel columns are most common words which allrady finded in step 2
##                   step 4:: Now, remove stop words (means ['the, in, is, are, of, and, etc...'])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english') # number of words, remove stop word = language

vecors = cv.fit_transform(df['tag']).toarray()
vecors
vecors[0]

print(cv.get_feature_names_out())


## Now find similarity between two or movie using cosin method
## so import sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vecors)
similarity[1]

# m = df[df['title'] == 'Avatar'].index[0]
# s = similarity[m]
# sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6]


def recommend(movie):
    movie_index = df[df['title'] == movie].index[0] # to find movie index number
    distances = similarity[movie_index] # to find similar movie which we find
    # show movie_id with reverse first 5 similer movies
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6] 

    for i in movie_list:
        # print(i[0])     # print movie id
        print(df.iloc[i[0]].title) # print title(movie) name 

# returen movie
# recommend('Batman')