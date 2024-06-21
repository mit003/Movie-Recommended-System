import streamlit as st
import recommender as rd
import requests


def Fetch_poster(movie_id):
    # https://api.themoviedb.org/3/movie/19995?api_key=16601fbe8f836d9f8dc453b6e150920f
    # https://api.themoviedb.org/3/movies/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=16601fbe8f836d9f8dc453b6e150920f'.format(movie_id))
    data = response.json()
    # st.text(data)
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']



## recommend movies which similar like searched movie
def recommend(movie):
    movie_index = rd.df[rd.df['title'] == movie].index[0] # to find movie index number
    distances = rd.similarity[movie_index] # to find similar movie which we find

    # show movie_id with reverse first 5 similer movies
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6] 


    list_movie = []
    recommend_movies_poster = []
    for i in movie_list:
        movie_id = rd.df.iloc[i[0]].movie_id
        print(movie_id)     # print movie id
        list_movie.append(rd.df.iloc[i[0]].title) # print title(movie) name 
        recommend_movies_poster.append(Fetch_poster(movie_id)) # get movie poster and append in list i[0] is movie id


    return list_movie, recommend_movies_poster


st.title('Movie Recommendwer System')

movies_name = rd.df['title']
# print(movies_name)

selected_movie = st.selectbox('How would you like...', movies_name)

# list out similer movies like as selected movies
'''if st.button("Recommender1"):
    recommendtions = recommend(selected_movie)
    for i in recommendtions:
         st.write(i)'''

        
if st.button("Recommender"):
    name, poster = recommend(selected_movie)
    col = st.columns(5)
    for i, (name_item, poster_item) in enumerate(zip(name, poster)):  # Use enumerate to get both index and value
        with col[i]:  # Use 'with' statement to set the current column
            st.text(name_item)
            st.image(poster_item, width=220)  # Adjust the width as needed
            
            
