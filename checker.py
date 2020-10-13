import pandas as pd
df=pd.read_csv('articles.csv',sep=',',index_col=False)
assert sorted(list(df['Link']))==sorted(list(set(list(df['Link']))))