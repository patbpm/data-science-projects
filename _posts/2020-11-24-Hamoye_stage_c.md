---
title: "Data Hamayo Project"
date: 2020-11-24
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---


**Importing Libraries**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
import scipy.stats
```
  ```
    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
  ```



**Importing the Data**


```python

#Data of the africa food production
url = 'https://raw.githubusercontent.com/patbpm/Hamoye_quiz/master/Africa%20Food%20Production%20(2004%20-%202013).csv'

africa_food_prod_data_df = pd.read_csv(url)
```

**DATA EDA**


```python
africa_food_prod_data_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Item</th>
      <th>Year</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2004</td>
      <td>2731</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2005</td>
      <td>2415</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2006</td>
      <td>2688</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2007</td>
      <td>2319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2008</td>
      <td>1111</td>
    </tr>
  </tbody>
</table>
</div>




```python
africa_food_prod_data_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>23110.000000</td>
      <td>23110.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2008.498269</td>
      <td>327.785201</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.871740</td>
      <td>1607.940343</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2004.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2006.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2008.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2011.000000</td>
      <td>108.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2013.000000</td>
      <td>54000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
africa_food_prod_data_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23110 entries, 0 to 23109
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   Country  23110 non-null  object
     1   Item     23110 non-null  object
     2   Year     23110 non-null  int64 
     3   Value    23110 non-null  int64 
    dtypes: int64(2), object(2)
    memory usage: 722.3+ KB
    


```python
africa_food_prod_data_df.columns
```



  ```
    Index(['Country', 'Item', 'Year', 'Value'], dtype='object')
  ```



```python
africa_food_prod_data_df.shape
```



  
    (23110, 4)




```python
africa_food_prod_data_df.isnull().sum()
```




    Country    0
    Item       0
    Year       0
    Value      0
    dtype: int64




```python
#Grouping Dataset

africa_food_prod_data_df.groupby(["Country", "Year"])["Value"].sum()
grouped_africa_food_prod_data_df = pd.DataFrame(africa_food_prod_data_df)
grouped_africa_food_prod_data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Item</th>
      <th>Year</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2004</td>
      <td>2731</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2005</td>
      <td>2415</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2006</td>
      <td>2688</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2007</td>
      <td>2319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Algeria</td>
      <td>Wheat and products</td>
      <td>2008</td>
      <td>1111</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23105</th>
      <td>Zimbabwe</td>
      <td>Crustaceans</td>
      <td>2009</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23106</th>
      <td>Zimbabwe</td>
      <td>Crustaceans</td>
      <td>2010</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23107</th>
      <td>Zimbabwe</td>
      <td>Crustaceans</td>
      <td>2011</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23108</th>
      <td>Zimbabwe</td>
      <td>Crustaceans</td>
      <td>2012</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23109</th>
      <td>Zimbabwe</td>
      <td>Crustaceans</td>
      <td>2013</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>23110 rows × 4 columns</p>
</div>




```python
grouped_africa_food_prod_data_df.shape
```




    (23110, 4)




```python
# The plot will show the trend in food production between 2004 and 2013

sns.set(rc = {'figure.figsize':(15,14)})
trend_grouped_africa_food_prod_data_df = sns.lineplot(x=grouped_africa_food_prod_data_df["Year"], y=grouped_africa_food_prod_data_df["Value"], hue=grouped_africa_food_prod_data_df["Country"], palette = 'bright', ci = None, data=grouped_africa_food_prod_data_df)
trend_grouped_africa_food_prod_data_df.set_yscale('log')
trend_grouped_africa_food_prod_data_df.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
trend_grouped_africa_food_prod_data_df.set(Title =  'Food Producing Countries')
```




    [Text(0.5, 1.0, 'Food Producing Countries')]




    
![png](({{ site.url }}{{ site.baseurl }}/images/Hamoye_stage_c/Hamoye_stage_c_15_1.png)
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)
    


 The graphs shows clearely that Nigeria, Egypt and South Africa have consistently, respectively been the top three food producing countries in Africa


```
# This Map View Food Producting Countries
```
