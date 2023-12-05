# %%
# get the data from my own sql database,
# which is in 'df.db'
import pandas as pd
import numpy as np

import sqlite3

conn= sqlite3.connect('News_Article_2016.sqlite') #'df.db')

# what are the tables in this database?
df= pd.read_sql_query('''
    SELECT name FROM sqlite_master 
    WHERE type='table' 
    ORDER BY name''', 
    conn)
tblName= df['name'][0]
tblName

df= pd.read_sql_query(f'''
        SELECT * FROM {tblName}
        -- limit 1000
        ''', conn)
df  
# %%
# estimate the distinct number of names 
# in `Name` column

# %%
# estimate the distinct number of names
# in `Name` column
df['Name'].nunique()
# %%
# list the distinct names in `Name` column
# sorted by alphabet
df['Name'].unique()
# %%
# select the rows with `Name` column
# equal to one of the following names
# ['Apple', 'Microsoft', 'Google', 'Amazon', 'Nvidia']
# ignore the case

# %%
# select the rows with `Name` column
# equal to one of the following names
# ['Apple', 'Microsoft', 'Google', 'Amazon', 'Nvidia']
# ignore the case
df0= df[df['Name'].str.lower().isin(['apple', 'microsoft', 'google', 'amazon', 'nvidia'])]
df0

# %%
#df0.to_excel('df0.xlsx', index= False)

# %%
'''
As of 2023, the top 10 companies in the United States by market capitalization are:

Apple               (AAPL):             $2.974 trillion
Microsoft           (MSFT):             $2.783 trillion
Alphabet            (Google) (GOOG):    $1.659 trillion
Amazon              (AMZN):             $1.519 trillion
NVIDIA              (NVDA):             $1.155 trillion
Meta Platforms      (Facebook) (META):  $834.74 billion
Berkshire Hathaway  (BRK-B):            $776.75 billion
Tesla               (TSLA):             $759.22 billion
Eli Lilly           (LLY):              $554.43 billion
Visa                (V):                $526.57 billion​​.
'''

'''
In 2016, the top 10 companies by market capitalization were:

Apple:              $617.59 billion
Alphabet (Google):  $531.97 billion
Microsoft:          $483.16 billion
ExxonMobil:         $404.39 billion
Berkshire Hathaway: $374.28 billion
Johnson & Johnson:  $356.31 billion
General Electric:   $313.43 billion
Amazon:             $308.77 billion
Facebook:           $279.55 billion
Wells Fargo:        $276.78 billion​​.
'''
#%%
theTop10= [
'Apple',             
'Google', 
'Microsoft',         
'Exxon Mobil',        
'Berkshire Hathaway',
'Johnson & Johnson', 
'General Electric',  
'Amazon',            
'Facebook',          
'Wells Fargo']

#
sorted(theTop10)


# %%
df10= df[df['Name'].str.lower().isin([x.lower() for x in theTop10])]
#df10.to_excel('df10.xlsx', index= False)

# %%
# find the number of distinct names in `Name` column
# in the top 10 companies
df10['Name'].nunique()
# %%
# list the distinct names in `Name` column
# in the top 10 companies
df10['Name'].unique()

# %%
# using 'LENGTH: \d+ words' 
# to extract the length of the article
# in the `cont` column and split it into 2 columns
# `cont2` and `len`
# if the pattern is not found,
# then assign the whole string to `cont2` and `len` is NaN

# %%
# using 'LENGTH:\s*\d+\s*words'
# to extract the length of the article
# in the `cont` column and split it into 2 columns
# `head`, `cont2` 
# keeping the pattern and put it in `head`
# if the pattern is not found,
# then assign the whole string to `cont2` and `head` is Null

# %%
# using 'LENGTH:\s*\d+\s*words'
# to extract the length of the article
# in the `cont` column and split it into 2 columns
# `head`, `cont2`
# keeping the pattern and put it in `head`
# if the pattern is not found,
# then assign the whole string to `cont2` and `head` is Null

import re

#df10['head']= df10['cont'].apply(lambda x: re.split(r'(LENGTH:\s*\d+\s*words)', x, 1)[0].strip())
#df10['cont2']= df10['cont'].apply(lambda x: re.split(r'LENGTH:\s*\d+\s*words', x, 1)[-1].strip())

#df10['len']= df10['cont'].apply(lambda x: re.search('LENGTH:\s*\d+\s*words', x, 1).group(0).split()[-2])

# %%

#%%
df10

#%%
#df10.to_excel('df10.xlsx', index= False)



# %%
'''
Top 25 Banks. 
Name Ticker 

00, 'ACE' 	, 'ACE LIMITED', 
01, 'AFL' 	, 'A F L A C INC', 'AFLAC',
02, 'AIG' 	, 'AMERICAN INTERNATIONAL GROUP INC', 'AMERICAN INTERNATIONAL',
03, 'AMT' 	, 'AMERICAN TOWER CORPORATION', 'AMERICAN TOWER',
04, 'AXP' 	, 'AMERICAN EXPRESS CO', 'AMERICAN EXPRESS',
05, 'BAC' 	, 'BANK OF AMERICA CORP', 'BANK OF AMERICA',
06, 'BBT' 	, 'TRUIST FINANCIAL CORP', 
07, 'BEN' 	, 'FRANKLIN RESOURCES INC', 
08, 'BK'	, 'BANK NEW YORK INC', 
09, 'BLK' 	, 'BLACKROCK INC', 
10, 'BRK.B1', 'BERKSHIRE HATHAWAY INC DEL', 
11, 'C' 	, 'CITIGROUP INC', 
12, 'CB' 	, 'CHUBB LIMITED', 
13, 'COF' 	, 'CAPITAL ONE FINANCIAL CORP', 
14, 'HCP' 	, 'HEALTH CARE PROPERTY INVESTORS INC', 
15, 'JPM' 	, 'JPMORGAN CHASE & CO', 
16, 'MET' 	, 'METLIFE INC', 
17, 'PNC' 	, 'P N C FINANCIAL SERVICES GRP INC', 
18, 'PRU' 	, 'PRUDENTIAL FINANCIAL INC', 
19, 'PSA' 	, 'PUBLIC STORAGE (PSA)', 
20, 'SPG' 	, 'SIMON PROPERTY GROUP INC NEW', 
21, 'STT' 	, 'STATE STREET CORPORATION (STT)', 
22, 'TRV' 	, 'ST PAUL TRAVELERS COS INC', 
23, 'USB' 	, 'U S BANCORP DEL', 
24, 'WFC' 	, 'WELLS FARGO & CO NEW', 

Note that ACE and CB merged on January 14, 2016. Hence they are two 
different insurance companies in our sample. BBT was formerly known as 
Bankers Trust and merged with SunTrust Bank in December 2019.
'''


top25Banks= (
    'ACE LTD', 
    'AFLAC',
    'AMERICAN INTERNATIONAL',
    'AMERICAN TOWER',
    'AMERICAN EXPRESS',
    'BANK OF AMERICA',
    'TRUIST FINANCIAL', 
    'FRANKLIN RESOURCES', 
    'BANK OF NEW YORK', 
    'BLACKROCK', 
    'BERKSHIRE HATHAWAY', 
    'CITIGROUP', 
    'CHUBB LTD', 
    'CAPITAL ONE FINANCIAL', 
    'HEALTH CARE PROPERTY INVESTORS', 
    'JPMORGAN CHASE', 
    'METLIFE', 
    'PNC FINANCIAL', 
    'PRUDENTIAL FINANCIAL', 
    'PUBLIC STORAGE', 
    'SIMON PROPERTY', 
    'STATE STREET', 
    'TRAVELERS COS', 
    'U S BANCORP', 
    'WELLS FARGO') 
#%%
dfL= []
for bank in top25Banks:
    sqlQuery= f'''
        SELECT * FROM {tblName}
        WHERE Name LIKE '%{bank}%'
        '''
    df1= pd.read_sql_query(sqlQuery, conn)
    if df1.empty:
        print(f'"{bank}" not found')
    dfL.append(df1)

# 有3家銀行没有找到
'''
"ACE LTD" not found                         # ACE LTD merged with CHUBB LTD
"TRUIST FINANCIAL" not found                # TRUIST FINANCIAL is a new bank
"HEALTH CARE PROPERTY INVESTORS" not found  # HEALTH CARE PROPERTY INVESTORS is a REIT
'''
#%%
df2= pd.concat(dfL)
df2

# 7031 rows × 10 columns
# 已經有 7031 篇文章是有關於這(25-3=22)家銀行的文章
# 就先用這些文章來做分析
#

#df2.to_excel('df_top25Banks.xlsx', index= False)

    

# %%
# find the number of distinct names in `Name` column of df2
df2['Name'].nunique()
# %%
# list the distinct names in `Name` column of df2
df2['Name'].unique()
# %%
df2['Name'].unique()


# %%

# for each bank, find the number of articles
# and the average length of articles
#%%

# for each bank, find the number of articles

#%%
df2['Name'].value_counts()
#%%

# for each article, find the number of words
# in the article
df2['len']= df2['Content'].apply(lambda x: len(x.split()))
df2
# for each bank, find the total number of words in the articles
# and the average number of words in the articles
df2.groupby('Name').agg({'len': ['sum', 'mean']})


# %%
# for each bank, find the number of articles
# and the average length of articles, and the total number of words
# put them in a dataframe
df3= df2.groupby('Name').agg({'len': ['count', 'sum', 'mean' ]})
df3.columns= ['numArticles', 'totalWords', 'avgWords']
df3

#%%
'''
	numArticles	totalWords	avgWords
Name			
AFLAC	258	196462	761.480620
AMERICAN EXPRESS	359	139449	388.437326
AMERICAN INTERNATIONAL	395	173170	438.405063
AMERICAN TOWER	321	490471	1527.947040
BANK OF AMERICA	372	127901	343.819892
BANK OF NEW YORK MELLON	317	214611	677.006309
BERKSHIRE HATHAWAY	380	171844	452.221053
BLACKROCK	428	214568	501.327103
CAPITAL ONE FINANCIAL	221	117638	532.298643
CHUBB LTD	104	57894	556.673077
CITIGROUP	392	158523	404.395408
FRANKLIN RESOURCES	289	165884	573.993080
JPMORGAN CHASE	257	121009	470.852140
METLIFE	502	288828	575.354582
PNC FINANCIAL SVCS	3	3558	1186.000000
PRUDENTIAL FINANCIAL	412	219563	532.919903
PUBLIC STORAGE	153	271402	1773.869281
SIMON PROPERTY	374	306107	818.467914
STATE STREET	396	160222	404.601010
TRAVELERS COS	315	263325	835.952381
U S BANCORP	335	278873	832.456716
WELLS FARGO	448	170842	381.343750
'''

'''
	numArticles	totalWords	avgWords
Name			
AFLAC	258	227729	882.670543
AMERICAN EXPRESS	359	149943	417.668524
AMERICAN INTERNATIONAL	395	186728	472.729114
AMERICAN TOWER	321	500388	1558.841121
BANK OF AMERICA	372	139341	374.572581
BANK OF NEW YORK MELLON	317	225299	710.722397
BERKSHIRE HATHAWAY	380	183702	483.426316
BLACKROCK	428	232159	542.427570
CAPITAL ONE FINANCIAL	221	127224	575.674208
CHUBB LTD	104	61073	587.240385
CITIGROUP	392	173887	443.589286
FRANKLIN RESOURCES	289	176181	609.622837
JPMORGAN CHASE	257	133175	518.190661
METLIFE	502	308699	614.938247
PNC FINANCIAL SVCS	3	3641	1213.666667
PRUDENTIAL FINANCIAL	412	236046	572.927184
PUBLIC STORAGE	153	276033	1804.137255
SIMON PROPERTY	374	325396	870.042781
STATE STREET	396	175533	443.265152
TRAVELERS COS	315	273764	869.092063
U S BANCORP	335	290064	865.862687
WELLS FARGO	448	188673	421.145089
'''
#%%
# save df3 to excel as a newsheet in the same file
# with the name 'statistics_of_articles'

#with pd.ExcelWriter('df_top25Banks.xlsx', mode='a', engine='openpyxl') as writer:
#    df3.to_excel(writer, sheet_name='statistics_of_articles')


# %%
'''
22 間銀行的文章總數量= 7031
每間銀行的文章數量不一樣，從3篇到502篇不等，
平均數是 319.59 篇，中位數是 321 篇，標準差是 118.67 篇。
最多的是 METLIFE 有 502 篇，最少的是 PNC FINANCIAL SVCS 有 3 篇。

每一間銀行的文章平均長度不一樣，從 343.82 字到 1773.87 字不等，
平均數是 701.27 字，中位數是 532.92 字，標準差是 361.15 字。
最多的是 PUBLIC STORAGE 有 1773.87 字，最少的是 BANK OF AMERICA 有 343.82 字。

每一間銀行的文章總字數不一樣，從 3558 字到 490471 字不等，
平均數是 190,000 字，中位數是 171,844 字，標準差是 112,000 字。
最多的是 AMERICAN TOWER 有 490471 字，最少的是 PNC FINANCIAL SVCS 有 3558 字。
'''

'''
我想把這些文章用來做 word2vec 的模型，
以及用來做 doc2vec 的模型。
並進而用這些模型來做一些分析。
找出這些銀行的相關性。
用 3d 圖表來呈現。
'''

# %%
import re

def clean_article_content(text):
    """
    Clean the article content by removing non-article information such as title, length, etc.
    This function uses regular expressions to identify and remove common patterns.
    """
    # Example pattern to remove (This can be modified based on the actual data structure)
    # Removing patterns like "LANGUAGE: ENGLISH  PUBLICATION-TYPE: Newsletter..."
    cleaned_text= re.sub(r'LANGUAGE:.*?PUBLICATION-TYPE:.*?\n', '', text, flags=re.DOTALL)

    # Additional cleaning rules can be added here based on observed patterns

    return cleaned_text.strip()

# Applying the cleaning function to a sample of the dataframe
df_sample_cleaned= df['Content'].sample(10).apply(
    clean_article_content)
df_sample_cleaned



# %%
x= df_sample_cleaned.iloc[0]
# %%
len(x.split())
# %%
print(x)
# %%
#%%
def extract_and_clean_article_content(text):
    """
    Extract the article length information and clean the article content.
    The function finds the 'LENGTH: \d+ words' pattern and extracts it,
    then removes non-article information from the text.
    """
    # Extracting article length information
    length_info = re.search(r'LENGTH:\s+(\d+)\s+words', text)
    article_length = length_info.group(1) if length_info else 'Unknown'

    # Cleaning the text by removing the extracted pattern and other non-article information
    cleaned_text= re.sub(r'LENGTH:\s+\d+\s+words', '', text)  # Removing the length pattern
    cleaned_text= re.sub(r'LANGUAGE:.*?[A-Z\-]+:', '', 
                          cleaned_text, flags=re.DOTALL)  # Additional cleaning
    cleaned_text= re.sub(r'PUBLICATION-TYPE:.*?[A-Z\-]+:', '', 
                          cleaned_text, flags=re.DOTALL)  # Additional cleaning
    cleaned_text= re.sub(r'BYLINE:.*?[A-Z\-]+:', '', 
                          cleaned_text, flags=re.DOTALL)  # Additional cleaning
    cleaned_text= re.sub(r'\d{1,2}:\d{1,2} AM EST\s*', '', 
                          cleaned_text, flags=re.DOTALL)  # Additional cleaning


    return article_length, cleaned_text.strip()

# Applying the updated cleaning function to a sample of the dataframe
df_cleaned= df2['Content'].apply(extract_and_clean_article_content)

df2['cleaned']= df_cleaned.apply(lambda x: x[1])
df2['len']= df_cleaned.apply(lambda x: x[0])

#df2['len'] should be integer
df2['len']= df2['len'].apply(lambda x: int(x) if x.isdigit() else np.nan)

df2
# %%

#%%
#df2.to_excel('df2_debug.xlsx', index= False)
df2.to_excel('df_top25Banks_cleaned.xlsx', index= False)
# %%

df2.sample(100).to_csv('df_sample100.csv', index= False)
# %%
df2[['Name', 'cleaned']].sample(100).to_excel('df_sample100.xlsx', index= False)

# %%

import pandas as pd
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Load the data
#df = pd.read_excel('path_to_your_file.xlsx')

df= df2[['Name', 'cleaned']] #.sample(1000)

# Preprocess the text
def preprocess_text(text):
    return word_tokenize(text.lower())

df['processed'] = df['cleaned'].apply(preprocess_text)

# Train the Word2Vec model
model_w2v= Word2Vec(
    sentences=df['processed'], 
    vector_size=100, 
    window=5, 
    min_count=1, 
    workers=4)

# Prepare data for Doc2Vec
tagged_data= [
    TaggedDocument(words=_d, tags=[str(i)]) 
    for i, _d in enumerate(df['processed'])]

# Train the Doc2Vec model
model_d2v= Doc2Vec(
    tagged_data, 
    vector_size=100, 
    window=5, 
    min_count=1, 
    workers=4)

# Now you can use model_w2v and model_d2v for various analyses

# %%

# Example: Find words similar to 'bank'
similar_words= model_w2v.wv.most_similar('bank')
similar_words

for w in ['bank', 
          'company',
          'finance',
          'risk',
          'interest',
          'return', 
          'stock', 
          'bond']:
    similar_words= model_w2v.wv.most_similar(w)
    print(f'{w= }, {similar_words= }')
# %%
# Example: Vector for the word 'bank
word_vector= model_w2v.wv['bank']
word_vector
#%%
vocab= list(model_w2v.wv.index_to_key)
len(vocab), vocab
# %%
# Example: Infer the vector of a new document
new_doc= preprocess_text("Your new document text here.")
inferred_vector = model_d2v.infer_vector(new_doc)
inferred_vector

# %%
# Example: Find similar documents
similar_docs = model_d2v.dv.most_similar([inferred_vector])
similar_docs

# %%
df
# %%
doc= df['processed'].iloc[0]
#doc= preprocess_text(doc)
doc

# %%
#similar_docs= model_d2v.dv.most_similar(doc)
#similar_docs

# %%

#doc= df['processed'].iloc[0]
i= 1000
name= df['Name'].iloc[i]
doc=  df['cleaned'].iloc[i]
doc=  preprocess_text(doc)
inferred_vector= model_d2v.infer_vector(doc)
similar_docs= model_d2v.dv.most_similar([inferred_vector])
similar_names= [(j,df['Name'].iloc[j]) for j in [int(x[0]) for x in similar_docs]]

(i, name), similar_names

# %%
# Assuming 'df' is your dataframe and 'cleaned' is the column with text data
all_doc_vectors= [
    model_d2v.infer_vector(preprocess_text(doc)) 
    for doc in df['cleaned']]


# %%

from sklearn.manifold import TSNE

# Using t-SNE for dimensionality reduction
tsne_model= TSNE(n_components=2, random_state=0)  # Use n_components=3 for 3D
reduced_vectors= tsne_model.fit_transform(
    np.array(all_doc_vectors)
    )

# %%
import matplotlib.pyplot as plt

nNames= 1000

# Scatter plot for the reduced vectors
plt.figure(figsize=(12, 12))
plt.scatter(
    reduced_vectors[0:nNames, 0], 
    reduced_vectors[0:nNames, 1])

# Optionally, you can annotate points with bank names or indices
for i, name in enumerate(df['Name']):
    if i >= nNames:
        break
    plt.annotate(name, (reduced_vectors[i, 0], 
                        reduced_vectors[i, 1]))

plt.show()

# %%
# using 3-dimension to visualize the data
from mpl_toolkits.mplot3d import Axes3D

tsne_model_3d= TSNE(n_components=3, random_state=0)  # Use n_components=3 for 3D
reduced_vectors_3d= tsne_model_3d.fit_transform(
    np.array(all_doc_vectors)
    )

# %%
# Scatter plot for the reduced vectors
fig= plt.figure(figsize=(12, 12))
ax= fig.add_subplot(111, projection='3d')
ax.scatter(
    reduced_vectors_3d[0:nNames, 0], 
    reduced_vectors_3d[0:nNames, 1],
    reduced_vectors_3d[0:nNames, 2])

# Optionally, you can annotate points with bank names or indices
for i, name in enumerate(df['Name']):
    if i >= nNames:
        break
    ax.text(
        reduced_vectors_3d[i, 0], 
        reduced_vectors_3d[i, 1],
        reduced_vectors_3d[i, 2],
        name)
    
plt.show()

# %%
# using interactive 3-dimension to visualize the data
import plotly.express as px

fig = px.scatter_3d(
    reduced_vectors_3d[0:nNames], 
    x=0, y=1, z=2, 
    text=df['Name'][0:nNames])

fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='3D Scatter Plot'
)

fig.show()



# %%
