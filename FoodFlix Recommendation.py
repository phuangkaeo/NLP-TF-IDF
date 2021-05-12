import streamlit as st
import pandas as pd
from IPython.display import Image, HTML
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#Lodad Data
st.title("FoodFlix Recommendation")

food = pd.read_csv("Data/en.openfoodfacts.org.products.tsv", sep='\t', low_memory=False, nrows = 10000)

#
foodfact = food[['code','product_name','ingredients_text','stores','nutrition_grade_fr']]
#
# st.write(foodfact)
#
# foodfact.isna().sum()
#
foodfact.fillna('', inplace = True)
#


#
foodfact['product_name'] = foodfact['product_name'].astype('str')
foodfact['ingredients_text'] = foodfact['ingredients_text'].astype('str')
foodfact['stores'] = foodfact['stores'].astype('str')
foodfact['nutrition_grade_fr'] = foodfact['nutrition_grade_fr'].astype('str')
#
product_name_corpus = ' '.join(foodfact['product_name'])
ingredient_corpus = ' '.join(foodfact['ingredients_text'])
stores_corpus = ' '.join(foodfact['stores'])
nutrition_corpus = ' '.join(foodfact['nutrition_grade_fr'])

foodfact['content'] = foodfact[['product_name',
                                'ingredients_text',
                                'stores',
                                'nutrition_grade_fr']] .astype(str).apply(lambda x: ' // '.join(x),axis = 1)

foodfact['content'].fillna('Null', inplace = True)

st.write(foodfact)

tf = TfidfVectorizer(analyzer = 'word',
                     ngram_range = (1,2),
                     min_df = 0,
                     stop_words = 'english')
tfidf_matrix = tf.fit_transform(foodfact['content'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


results = {}
for idx, row in foodfact.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], foodfact['code'][i])
                    for i in similar_indices]
    results[row['code']] = similar_items[1:]

def item(id):
    name   = foodfact.loc[foodfact['code'] == id]['content'].tolist()[0].split(' // ')[0]
    ingredient   = ' \nIngredients: ' + foodfact.loc[foodfact['code'] == id]['content'].tolist()[0].split(' // ')[1][0:165] + '...'
    store   = ' \nStore: ' + foodfact.loc[foodfact['code'] == id]['content'].tolist()[0].split(' // ')[2][0:165] + '...'
    nutition   = ' \nNutrition Score: ' + foodfact.loc[foodfact['code'] == id]['content'].tolist()[0].split(' // ')[3][0:165] + '...'


    prediction = name  + ingredient + store + nutition
    return prediction

def recommend(item_id, num):
    print('Recommending ' + str(num) + ' products similar to ' + item(item_id))
    print('---')
    recs = results[item_id][:num]
    for rec in recs:
        print('\nRecommended: ' + item(rec[1]) + '\n(score:' + str(rec[0]) + ')')

# st.write(tf)
recommend(item_id=2929, num=5)