---
title: "Data Science Kaggle Survey 2020"
date: 2020-12-04
tags: [data wrangling, data science, EDA]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "data wrangling, data science, EDA"
mathjax: "true"
---
  

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```
```python
    /kaggle/input/kaggle-survey-2020/kaggle_survey_2020_responses.csv
    /kaggle/input/kaggle-survey-2020/supplementary_data/kaggle_survey_2020_methodology.pdf
    /kaggle/input/kaggle-survey-2020/supplementary_data/kaggle_survey_2020_answer_choices.pdf
    /kaggle/input/plotly-country-code-mapping/2014_world_gdp_with_codes.csv
```    

# **General Information**

In October 2020, Kaggle launched a big online survey for kagglers. There were multiple choice questions and some forms for open answers. 

Survey received 20,036 usable respondents from different countries and territories. 
In this kernel I will try to analyse this data and provide various insights. 

The Main tools that I will use in this kernel are Python as language and seaborn and plotly for visualisation.

I have decided to perform the analysis based on the country of the responders.


```python
# Import libraries
import numpy as np 
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', None)
from ipywidgets import interact, interactive, interact_manual
import ipywidgets as widgets
import colorlover as cl
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
#loading data
df = pd.read_csv(r'../input/kaggle-survey-2020/kaggle_survey_2020_responses.csv')

```

**List of questions**

Below are the list of all 35 questions asked during the survey


```python
question = df.iloc[0]
with pd.option_context("display.max_rows", 1000):
 for k, v in question.items():
    print(k, v)
 


```
```python
    Time from Start to Finish (seconds) Duration (in seconds)
    Q1 What is your age (# years)?
    Q2 What is your gender? - Selected Choice
    Q3 In which country do you currently reside?
    Q4 What is the highest level of formal education that you have attained or plan to attain within the next 2 years?
    Q5 Select the title most similar to your current role (or most recent title if retired): - Selected Choice
    Q6 For how many years have you been writing code and/or programming?
    Q7_Part_1 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python
    Q7_Part_2 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - R
    Q7_Part_3 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL
    Q7_Part_4 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C
    Q7_Part_5 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C++
    Q7_Part_6 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java
    Q7_Part_7 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Javascript
    Q7_Part_8 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Julia
    Q7_Part_9 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Swift
    Q7_Part_10 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Bash
    Q7_Part_11 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - MATLAB
    Q7_Part_12 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - None
    Q7_OTHER What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Other
    Q8 What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice
    Q9_Part_1 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Jupyter (JupyterLab, Jupyter Notebooks, etc) 
    Q9_Part_2 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  RStudio 
    Q9_Part_3 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio / Visual Studio Code 
    Q9_Part_4 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Click to write Choice 13
    Q9_Part_5 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  PyCharm 
    Q9_Part_6 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Spyder  
    Q9_Part_7 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Notepad++  
    Q9_Part_8 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Sublime Text  
    Q9_Part_9 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Vim / Emacs  
    Q9_Part_10 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  MATLAB 
    Q9_Part_11 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - None
    Q9_OTHER Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other
    Q10_Part_1 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Kaggle Notebooks
    Q10_Part_2 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Colab Notebooks
    Q10_Part_3 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Azure Notebooks
    Q10_Part_4 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Paperspace / Gradient 
    Q10_Part_5 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Binder / JupyterHub 
    Q10_Part_6 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Code Ocean 
    Q10_Part_7 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  IBM Watson Studio 
    Q10_Part_8 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon Sagemaker Studio 
    Q10_Part_9 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon EMR Notebooks 
    Q10_Part_10 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud AI Platform Notebooks 
    Q10_Part_11 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud Datalab Notebooks
    Q10_Part_12 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks Collaborative Notebooks 
    Q10_Part_13 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - None
    Q10_OTHER Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Other
    Q11 What type of computing platform do you use most often for your data science projects? - Selected Choice
    Q12_Part_1 Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - GPUs
    Q12_Part_2 Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - TPUs
    Q12_Part_3 Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - None
    Q12_OTHER Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - Other
    Q13 Approximately how many times have you used a TPU (tensor processing unit)?
    Q14_Part_1 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Matplotlib 
    Q14_Part_2 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Seaborn 
    Q14_Part_3 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Plotly / Plotly Express 
    Q14_Part_4 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 
    Q14_Part_5 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Shiny 
    Q14_Part_6 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3 js 
    Q14_Part_7 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Altair 
    Q14_Part_8 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Bokeh 
    Q14_Part_9 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Geoplotlib 
    Q14_Part_10 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Leaflet / Folium 
    Q14_Part_11 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - None
    Q14_OTHER What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other
    Q15 For how many years have you used machine learning methods?
    Q16_Part_1 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn 
    Q16_Part_2 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow 
    Q16_Part_3 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Keras 
    Q16_Part_4 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  PyTorch 
    Q16_Part_5 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Fast.ai 
    Q16_Part_6 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  MXNet 
    Q16_Part_7 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Xgboost 
    Q16_Part_8 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  LightGBM 
    Q16_Part_9 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  CatBoost 
    Q16_Part_10 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Prophet 
    Q16_Part_11 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  H2O 3 
    Q16_Part_12 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Caret 
    Q16_Part_13 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Tidymodels 
    Q16_Part_14 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  JAX 
    Q16_Part_15 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - None
    Q16_OTHER Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Other
    Q17_Part_1 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression
    Q17_Part_2 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests
    Q17_Part_3 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Gradient Boosting Machines (xgboost, lightgbm, etc)
    Q17_Part_4 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Bayesian Approaches
    Q17_Part_5 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Evolutionary Approaches
    Q17_Part_6 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Dense Neural Networks (MLPs, etc)
    Q17_Part_7 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Convolutional Neural Networks
    Q17_Part_8 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Generative Adversarial Networks
    Q17_Part_9 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Recurrent Neural Networks
    Q17_Part_10 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-3, etc)
    Q17_Part_11 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - None
    Q17_OTHER Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Other
    Q18_Part_1 Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - General purpose image/video tools (PIL, cv2, skimage, etc)
    Q18_Part_2 Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image segmentation methods (U-Net, Mask R-CNN, etc)
    Q18_Part_3 Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Object detection methods (YOLOv3, RetinaNet, etc)
    Q18_Part_4 Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)
    Q18_Part_5 Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Generative Networks (GAN, VAE, etc)
    Q18_Part_6 Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - None
    Q18_OTHER Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other
    Q19_Part_1 Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Word embeddings/vectors (GLoVe, fastText, word2vec)
    Q19_Part_2 Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Encoder-decorder models (seq2seq, vanilla transformers)
    Q19_Part_3 Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Contextualized embeddings (ELMo, CoVe)
    Q19_Part_4 Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Transformer language models (GPT-3, BERT, XLnet, etc)
    Q19_Part_5 Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - None
    Q19_OTHER Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other
    Q20 What is the size of the company where you are employed?
    Q21 Approximately how many individuals are responsible for data science workloads at your place of business?
    Q22 Does your current employer incorporate machine learning methods into their business?
    Q23_Part_1 Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions
    Q23_Part_2 Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data
    Q23_Part_3 Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas
    Q23_Part_4 Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows
    Q23_Part_5 Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models
    Q23_Part_6 Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning
    Q23_Part_7 Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work
    Q23_OTHER Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other
    Q24 What is your current yearly compensation (approximate $USD)?
    Q25 Approximately how much money have you (or your team) spent on machine learning and/or cloud computing services at home (or at work) in the past 5 years (approximate $USD)?
    Q26_A_Part_1 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Amazon Web Services (AWS) 
    Q26_A_Part_2 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure 
    Q26_A_Part_3 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Platform (GCP) 
    Q26_A_Part_4 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud / Red Hat 
    Q26_A_Part_5 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Oracle Cloud 
    Q26_A_Part_6 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  SAP Cloud 
    Q26_A_Part_7 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Salesforce Cloud 
    Q26_A_Part_8 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  VMware Cloud 
    Q26_A_Part_9 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Alibaba Cloud 
    Q26_A_Part_10 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Tencent Cloud 
    Q26_A_Part_11 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - None
    Q26_A_OTHER Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - Other
    Q27_A_Part_1 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Amazon EC2 
    Q27_A_Part_2 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  AWS Lambda 
    Q27_A_Part_3 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Elastic Container Service 
    Q27_A_Part_4 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Azure Cloud Services 
    Q27_A_Part_5 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure Container Instances 
    Q27_A_Part_6 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Azure Functions 
    Q27_A_Part_7 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Compute Engine 
    Q27_A_Part_8 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Functions 
    Q27_A_Part_9 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Run 
    Q27_A_Part_10 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud App Engine 
    Q27_A_Part_11 Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - No / None
    Q27_A_OTHER Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - Other
    Q28_A_Part_1 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon SageMaker 
    Q28_A_Part_2 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Forecast 
    Q28_A_Part_3 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Rekognition 
    Q28_A_Part_4 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Azure Machine Learning Studio 
    Q28_A_Part_5 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Azure Cognitive Services 
    Q28_A_Part_6 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud AI Platform / Google Cloud ML Engine
    Q28_A_Part_7 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Video AI 
    Q28_A_Part_8 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Natural Language 
    Q28_A_Part_9 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Vision AI 
    Q28_A_Part_10 Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice - No / None
    Q28_A_OTHER Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice - Other
    Q29_A_Part_1 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL 
    Q29_A_Part_2 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - PostgresSQL 
    Q29_A_Part_3 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - SQLite 
    Q29_A_Part_4 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Oracle Database 
    Q29_A_Part_5 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - MongoDB 
    Q29_A_Part_6 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Snowflake 
    Q29_A_Part_7 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - IBM Db2 
    Q29_A_Part_8 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft SQL Server 
    Q29_A_Part_9 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Access 
    Q29_A_Part_10 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Azure Data Lake Storage 
    Q29_A_Part_11 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon Redshift 
    Q29_A_Part_12 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon Athena 
    Q29_A_Part_13 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon DynamoDB 
    Q29_A_Part_14 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud BigQuery 
    Q29_A_Part_15 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud SQL 
    Q29_A_Part_16 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud Firestore 
    Q29_A_Part_17 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - None
    Q29_A_OTHER Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Other
    Q30 Which of the following big data products (relational database, data warehouse, data lake, or similar) do you use most often? - Selected Choice
    Q31_A_Part_1 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon QuickSight
    Q31_A_Part_2 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Power BI
    Q31_A_Part_3 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Google Data Studio
    Q31_A_Part_4 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Looker
    Q31_A_Part_5 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Tableau
    Q31_A_Part_6 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Salesforce
    Q31_A_Part_7 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Einstein Analytics
    Q31_A_Part_8 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Qlik
    Q31_A_Part_9 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Domo
    Q31_A_Part_10 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - TIBCO Spotfire
    Q31_A_Part_11 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Alteryx 
    Q31_A_Part_12 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Sisense 
    Q31_A_Part_13 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - SAP Analytics Cloud 
    Q31_A_Part_14 Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - None
    Q31_A_OTHER Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Other
    Q32 Which of the following business intelligence tools do you use most often? - Selected Choice
    Q33_A_Part_1 Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)
    Q33_A_Part_2 Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)
    Q33_A_Part_3 Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)
    Q33_A_Part_4 Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)
    Q33_A_Part_5 Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)
    Q33_A_Part_6 Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)
    Q33_A_Part_7 Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - No / None
    Q33_A_OTHER Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Other
    Q34_A_Part_1 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud AutoML 
    Q34_A_Part_2 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H20 Driverless AI  
    Q34_A_Part_3 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks AutoML 
    Q34_A_Part_4 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  DataRobot AutoML 
    Q34_A_Part_5 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Tpot 
    Q34_A_Part_6 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto-Keras 
    Q34_A_Part_7 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto-Sklearn 
    Q34_A_Part_8 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto_ml 
    Q34_A_Part_9 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Xcessiv 
    Q34_A_Part_10 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   MLbox 
    Q34_A_Part_11 Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - No / None
    Q34_A_OTHER Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other
    Q35_A_Part_1 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Neptune.ai 
    Q35_A_Part_2 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Weights & Biases 
    Q35_A_Part_3 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Comet.ml 
    Q35_A_Part_4 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Sacred + Omniboard 
    Q35_A_Part_5 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  TensorBoard 
    Q35_A_Part_6 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Guild.ai 
    Q35_A_Part_7 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Polyaxon 
    Q35_A_Part_8 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Trains 
    Q35_A_Part_9 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Domino Model Monitor 
    Q35_A_Part_10 Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice - No / None
    Q35_A_OTHER Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice - Other
    Q36_Part_1 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Plotly Dash 
    Q36_Part_2 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Streamlit 
    Q36_Part_3 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  NBViewer 
    Q36_Part_4 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  GitHub 
    Q36_Part_5 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Personal blog 
    Q36_Part_6 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Kaggle 
    Q36_Part_7 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Colab 
    Q36_Part_8 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Shiny 
    Q36_Part_9 Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice - I do not share my work publicly
    Q36_OTHER Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice - Other
    Q37_Part_1 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera
    Q37_Part_2 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX
    Q37_Part_3 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Learn Courses
    Q37_Part_4 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp
    Q37_Part_5 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.ai
    Q37_Part_6 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity
    Q37_Part_7 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy
    Q37_Part_8 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - LinkedIn Learning
    Q37_Part_9 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Cloud-certification programs (direct from AWS, Azure, GCP, or similar)
    Q37_Part_10 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - University Courses (resulting in a university degree)
    Q37_Part_11 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - None
    Q37_OTHER On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Other
    Q38 What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice
    Q39_Part_1 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)
    Q39_Part_2 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Email newsletters (Data Elixir, O'Reilly Data & AI, etc)
    Q39_Part_3 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, etc)
    Q39_Part_4 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (notebooks, forums, etc)
    Q39_Part_5 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, Coursera forums, etc)
    Q39_Part_6 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Kaggle YouTube, Cloud AI Adventures, etc)
    Q39_Part_7 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)
    Q39_Part_8 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Analytics Vidhya, etc)
    Q39_Part_9 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (peer-reviewed journals, conference proceedings, etc)
    Q39_Part_10 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)
    Q39_Part_11 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - None
    Q39_OTHER Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Other
    Q26_B_Part_1 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Amazon Web Services (AWS) 
    Q26_B_Part_2 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Microsoft Azure 
    Q26_B_Part_3 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Google Cloud Platform (GCP) 
    Q26_B_Part_4 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  IBM Cloud / Red Hat 
    Q26_B_Part_5 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Oracle Cloud 
    Q26_B_Part_6 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  SAP Cloud 
    Q26_B_Part_7 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  VMware Cloud 
    Q26_B_Part_8 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Salesforce Cloud 
    Q26_B_Part_9 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Alibaba Cloud 
    Q26_B_Part_10 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Tencent Cloud 
    Q26_B_Part_11 Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice - None
    Q26_B_OTHER Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice - Other
    Q27_B_Part_1 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Amazon EC2 
    Q27_B_Part_2 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  AWS Lambda 
    Q27_B_Part_3 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Amazon Elastic Container Service 
    Q27_B_Part_4 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Azure Cloud Services 
    Q27_B_Part_5 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Microsoft Azure Container Instances 
    Q27_B_Part_6 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Azure Functions 
    Q27_B_Part_7 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Google Cloud Compute Engine 
    Q27_B_Part_8 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Google Cloud Functions 
    Q27_B_Part_9 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Google Cloud Run 
    Q27_B_Part_10 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Google Cloud App Engine 
    Q27_B_Part_11 In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice - None
    Q27_B_OTHER In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice - Other
    Q28_B_Part_1 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Amazon SageMaker 
    Q28_B_Part_2 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Amazon Forecast 
    Q28_B_Part_3 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Amazon Rekognition 
    Q28_B_Part_4 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Azure Machine Learning Studio 
    Q28_B_Part_5 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Azure Cognitive Services 
    Q28_B_Part_6 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Google Cloud AI Platform / Google Cloud ML Engine
    Q28_B_Part_7 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Google Cloud Video AI 
    Q28_B_Part_8 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Google Cloud Natural Language 
    Q28_B_Part_9 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Google Cloud Vision AI 
    Q28_B_Part_10 In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice - None
    Q28_B_OTHER In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice - Other
    Q29_B_Part_1 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - MySQL 
    Q29_B_Part_2 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - PostgresSQL 
    Q29_B_Part_3 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - SQLite 
    Q29_B_Part_4 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Oracle Database 
    Q29_B_Part_5 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - MongoDB 
    Q29_B_Part_6 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Snowflake 
    Q29_B_Part_7 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - IBM Db2 
    Q29_B_Part_8 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Microsoft SQL Server 
    Q29_B_Part_9 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Microsoft Access 
    Q29_B_Part_10 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Microsoft Azure Data Lake Storage 
    Q29_B_Part_11 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Amazon Redshift 
    Q29_B_Part_12 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Amazon Athena 
    Q29_B_Part_13 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Amazon DynamoDB 
    Q29_B_Part_14 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Google Cloud BigQuery 
    Q29_B_Part_15 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Google Cloud SQL 
    Q29_B_Part_16 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Google Cloud Firestore 
    Q29_B_Part_17 Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - None
    Q29_B_OTHER Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Other
    Q31_B_Part_1 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Microsoft Power BI
    Q31_B_Part_2 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Amazon QuickSight
    Q31_B_Part_3 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Google Data Studio
    Q31_B_Part_4 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Looker
    Q31_B_Part_5 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Tableau
    Q31_B_Part_6 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Salesforce
    Q31_B_Part_7 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Einstein Analytics
    Q31_B_Part_8 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Qlik
    Q31_B_Part_9 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Domo
    Q31_B_Part_10 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - TIBCO Spotfire
    Q31_B_Part_11 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Alteryx 
    Q31_B_Part_12 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Sisense 
    Q31_B_Part_13 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - SAP Analytics Cloud 
    Q31_B_Part_14 Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - None
    Q31_B_OTHER Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Other
    Q33_B_Part_1 Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)
    Q33_B_Part_2 Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)
    Q33_B_Part_3 Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)
    Q33_B_Part_4 Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)
    Q33_B_Part_5 Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)
    Q33_B_Part_6 Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google Cloud AutoML, H20 Driverless AI)
    Q33_B_Part_7 Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - None
    Q33_B_OTHER Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Other
    Q34_B_Part_1 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -  Google Cloud AutoML 
    Q34_B_Part_2 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -  H20 Driverless AI  
    Q34_B_Part_3 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -  Databricks AutoML 
    Q34_B_Part_4 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -  DataRobot AutoML 
    Q34_B_Part_5 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Tpot 
    Q34_B_Part_6 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Auto-Keras 
    Q34_B_Part_7 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Auto-Sklearn 
    Q34_B_Part_8 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Auto_ml 
    Q34_B_Part_9 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Xcessiv 
    Q34_B_Part_10 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   MLbox 
    Q34_B_Part_11 Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - None
    Q34_B_OTHER Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Other
    Q35_B_Part_1 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Neptune.ai 
    Q35_B_Part_2 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Weights & Biases 
    Q35_B_Part_3 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Comet.ml 
    Q35_B_Part_4 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Sacred + Omniboard 
    Q35_B_Part_5 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  TensorBoard 
    Q35_B_Part_6 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Guild.ai 
    Q35_B_Part_7 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Polyaxon 
    Q35_B_Part_8 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Trains 
    Q35_B_Part_9 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Domino Model Monitor 
    Q35_B_Part_10 In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice - None
    Q35_B_OTHER In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice - Other
```    


```python
df_1 = df.drop([0])
df_1.shape
```



```
    (20036, 355)
```



```python
df_1.head(2)
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
      <th>Time from Start to Finish (seconds)</th>
      <th>Q1</th>
      <th>Q2</th>
      <th>Q3</th>
      <th>Q4</th>
      <th>Q5</th>
      <th>Q6</th>
      <th>Q7_Part_1</th>
      <th>Q7_Part_2</th>
      <th>Q7_Part_3</th>
      <th>Q7_Part_4</th>
      <th>Q7_Part_5</th>
      <th>Q7_Part_6</th>
      <th>Q7_Part_7</th>
      <th>Q7_Part_8</th>
      <th>Q7_Part_9</th>
      <th>Q7_Part_10</th>
      <th>Q7_Part_11</th>
      <th>Q7_Part_12</th>
      <th>Q7_OTHER</th>
      <th>Q8</th>
      <th>Q9_Part_1</th>
      <th>Q9_Part_2</th>
      <th>Q9_Part_3</th>
      <th>Q9_Part_4</th>
      <th>Q9_Part_5</th>
      <th>Q9_Part_6</th>
      <th>Q9_Part_7</th>
      <th>Q9_Part_8</th>
      <th>Q9_Part_9</th>
      <th>Q9_Part_10</th>
      <th>Q9_Part_11</th>
      <th>Q9_OTHER</th>
      <th>Q10_Part_1</th>
      <th>Q10_Part_2</th>
      <th>Q10_Part_3</th>
      <th>Q10_Part_4</th>
      <th>Q10_Part_5</th>
      <th>Q10_Part_6</th>
      <th>Q10_Part_7</th>
      <th>Q10_Part_8</th>
      <th>Q10_Part_9</th>
      <th>Q10_Part_10</th>
      <th>Q10_Part_11</th>
      <th>Q10_Part_12</th>
      <th>Q10_Part_13</th>
      <th>Q10_OTHER</th>
      <th>Q11</th>
      <th>Q12_Part_1</th>
      <th>Q12_Part_2</th>
      <th>Q12_Part_3</th>
      <th>Q12_OTHER</th>
      <th>Q13</th>
      <th>Q14_Part_1</th>
      <th>Q14_Part_2</th>
      <th>Q14_Part_3</th>
      <th>Q14_Part_4</th>
      <th>Q14_Part_5</th>
      <th>Q14_Part_6</th>
      <th>Q14_Part_7</th>
      <th>Q14_Part_8</th>
      <th>Q14_Part_9</th>
      <th>Q14_Part_10</th>
      <th>Q14_Part_11</th>
      <th>Q14_OTHER</th>
      <th>Q15</th>
      <th>Q16_Part_1</th>
      <th>Q16_Part_2</th>
      <th>Q16_Part_3</th>
      <th>Q16_Part_4</th>
      <th>Q16_Part_5</th>
      <th>Q16_Part_6</th>
      <th>Q16_Part_7</th>
      <th>Q16_Part_8</th>
      <th>Q16_Part_9</th>
      <th>Q16_Part_10</th>
      <th>Q16_Part_11</th>
      <th>Q16_Part_12</th>
      <th>Q16_Part_13</th>
      <th>Q16_Part_14</th>
      <th>Q16_Part_15</th>
      <th>Q16_OTHER</th>
      <th>Q17_Part_1</th>
      <th>Q17_Part_2</th>
      <th>Q17_Part_3</th>
      <th>Q17_Part_4</th>
      <th>Q17_Part_5</th>
      <th>Q17_Part_6</th>
      <th>Q17_Part_7</th>
      <th>Q17_Part_8</th>
      <th>Q17_Part_9</th>
      <th>Q17_Part_10</th>
      <th>Q17_Part_11</th>
      <th>Q17_OTHER</th>
      <th>Q18_Part_1</th>
      <th>Q18_Part_2</th>
      <th>Q18_Part_3</th>
      <th>Q18_Part_4</th>
      <th>Q18_Part_5</th>
      <th>Q18_Part_6</th>
      <th>Q18_OTHER</th>
      <th>Q19_Part_1</th>
      <th>Q19_Part_2</th>
      <th>Q19_Part_3</th>
      <th>Q19_Part_4</th>
      <th>Q19_Part_5</th>
      <th>Q19_OTHER</th>
      <th>Q20</th>
      <th>Q21</th>
      <th>Q22</th>
      <th>Q23_Part_1</th>
      <th>Q23_Part_2</th>
      <th>Q23_Part_3</th>
      <th>Q23_Part_4</th>
      <th>Q23_Part_5</th>
      <th>Q23_Part_6</th>
      <th>Q23_Part_7</th>
      <th>Q23_OTHER</th>
      <th>Q24</th>
      <th>Q25</th>
      <th>Q26_A_Part_1</th>
      <th>Q26_A_Part_2</th>
      <th>Q26_A_Part_3</th>
      <th>Q26_A_Part_4</th>
      <th>Q26_A_Part_5</th>
      <th>Q26_A_Part_6</th>
      <th>Q26_A_Part_7</th>
      <th>Q26_A_Part_8</th>
      <th>Q26_A_Part_9</th>
      <th>Q26_A_Part_10</th>
      <th>Q26_A_Part_11</th>
      <th>Q26_A_OTHER</th>
      <th>Q27_A_Part_1</th>
      <th>Q27_A_Part_2</th>
      <th>Q27_A_Part_3</th>
      <th>Q27_A_Part_4</th>
      <th>Q27_A_Part_5</th>
      <th>Q27_A_Part_6</th>
      <th>Q27_A_Part_7</th>
      <th>Q27_A_Part_8</th>
      <th>Q27_A_Part_9</th>
      <th>Q27_A_Part_10</th>
      <th>Q27_A_Part_11</th>
      <th>Q27_A_OTHER</th>
      <th>Q28_A_Part_1</th>
      <th>Q28_A_Part_2</th>
      <th>Q28_A_Part_3</th>
      <th>Q28_A_Part_4</th>
      <th>Q28_A_Part_5</th>
      <th>Q28_A_Part_6</th>
      <th>Q28_A_Part_7</th>
      <th>Q28_A_Part_8</th>
      <th>Q28_A_Part_9</th>
      <th>Q28_A_Part_10</th>
      <th>Q28_A_OTHER</th>
      <th>Q29_A_Part_1</th>
      <th>Q29_A_Part_2</th>
      <th>Q29_A_Part_3</th>
      <th>Q29_A_Part_4</th>
      <th>Q29_A_Part_5</th>
      <th>Q29_A_Part_6</th>
      <th>Q29_A_Part_7</th>
      <th>Q29_A_Part_8</th>
      <th>Q29_A_Part_9</th>
      <th>Q29_A_Part_10</th>
      <th>Q29_A_Part_11</th>
      <th>Q29_A_Part_12</th>
      <th>Q29_A_Part_13</th>
      <th>Q29_A_Part_14</th>
      <th>Q29_A_Part_15</th>
      <th>Q29_A_Part_16</th>
      <th>Q29_A_Part_17</th>
      <th>Q29_A_OTHER</th>
      <th>Q30</th>
      <th>Q31_A_Part_1</th>
      <th>Q31_A_Part_2</th>
      <th>Q31_A_Part_3</th>
      <th>Q31_A_Part_4</th>
      <th>Q31_A_Part_5</th>
      <th>Q31_A_Part_6</th>
      <th>Q31_A_Part_7</th>
      <th>Q31_A_Part_8</th>
      <th>Q31_A_Part_9</th>
      <th>Q31_A_Part_10</th>
      <th>Q31_A_Part_11</th>
      <th>Q31_A_Part_12</th>
      <th>Q31_A_Part_13</th>
      <th>Q31_A_Part_14</th>
      <th>Q31_A_OTHER</th>
      <th>Q32</th>
      <th>Q33_A_Part_1</th>
      <th>Q33_A_Part_2</th>
      <th>Q33_A_Part_3</th>
      <th>Q33_A_Part_4</th>
      <th>Q33_A_Part_5</th>
      <th>Q33_A_Part_6</th>
      <th>Q33_A_Part_7</th>
      <th>Q33_A_OTHER</th>
      <th>Q34_A_Part_1</th>
      <th>Q34_A_Part_2</th>
      <th>Q34_A_Part_3</th>
      <th>Q34_A_Part_4</th>
      <th>Q34_A_Part_5</th>
      <th>Q34_A_Part_6</th>
      <th>Q34_A_Part_7</th>
      <th>Q34_A_Part_8</th>
      <th>Q34_A_Part_9</th>
      <th>Q34_A_Part_10</th>
      <th>Q34_A_Part_11</th>
      <th>Q34_A_OTHER</th>
      <th>Q35_A_Part_1</th>
      <th>Q35_A_Part_2</th>
      <th>Q35_A_Part_3</th>
      <th>Q35_A_Part_4</th>
      <th>Q35_A_Part_5</th>
      <th>Q35_A_Part_6</th>
      <th>Q35_A_Part_7</th>
      <th>Q35_A_Part_8</th>
      <th>Q35_A_Part_9</th>
      <th>Q35_A_Part_10</th>
      <th>Q35_A_OTHER</th>
      <th>Q36_Part_1</th>
      <th>Q36_Part_2</th>
      <th>Q36_Part_3</th>
      <th>Q36_Part_4</th>
      <th>Q36_Part_5</th>
      <th>Q36_Part_6</th>
      <th>Q36_Part_7</th>
      <th>Q36_Part_8</th>
      <th>Q36_Part_9</th>
      <th>Q36_OTHER</th>
      <th>Q37_Part_1</th>
      <th>Q37_Part_2</th>
      <th>Q37_Part_3</th>
      <th>Q37_Part_4</th>
      <th>Q37_Part_5</th>
      <th>Q37_Part_6</th>
      <th>Q37_Part_7</th>
      <th>Q37_Part_8</th>
      <th>Q37_Part_9</th>
      <th>Q37_Part_10</th>
      <th>Q37_Part_11</th>
      <th>Q37_OTHER</th>
      <th>Q38</th>
      <th>Q39_Part_1</th>
      <th>Q39_Part_2</th>
      <th>Q39_Part_3</th>
      <th>Q39_Part_4</th>
      <th>Q39_Part_5</th>
      <th>Q39_Part_6</th>
      <th>Q39_Part_7</th>
      <th>Q39_Part_8</th>
      <th>Q39_Part_9</th>
      <th>Q39_Part_10</th>
      <th>Q39_Part_11</th>
      <th>Q39_OTHER</th>
      <th>Q26_B_Part_1</th>
      <th>Q26_B_Part_2</th>
      <th>Q26_B_Part_3</th>
      <th>Q26_B_Part_4</th>
      <th>Q26_B_Part_5</th>
      <th>Q26_B_Part_6</th>
      <th>Q26_B_Part_7</th>
      <th>Q26_B_Part_8</th>
      <th>Q26_B_Part_9</th>
      <th>Q26_B_Part_10</th>
      <th>Q26_B_Part_11</th>
      <th>Q26_B_OTHER</th>
      <th>Q27_B_Part_1</th>
      <th>Q27_B_Part_2</th>
      <th>Q27_B_Part_3</th>
      <th>Q27_B_Part_4</th>
      <th>Q27_B_Part_5</th>
      <th>Q27_B_Part_6</th>
      <th>Q27_B_Part_7</th>
      <th>Q27_B_Part_8</th>
      <th>Q27_B_Part_9</th>
      <th>Q27_B_Part_10</th>
      <th>Q27_B_Part_11</th>
      <th>Q27_B_OTHER</th>
      <th>Q28_B_Part_1</th>
      <th>Q28_B_Part_2</th>
      <th>Q28_B_Part_3</th>
      <th>Q28_B_Part_4</th>
      <th>Q28_B_Part_5</th>
      <th>Q28_B_Part_6</th>
      <th>Q28_B_Part_7</th>
      <th>Q28_B_Part_8</th>
      <th>Q28_B_Part_9</th>
      <th>Q28_B_Part_10</th>
      <th>Q28_B_OTHER</th>
      <th>Q29_B_Part_1</th>
      <th>Q29_B_Part_2</th>
      <th>Q29_B_Part_3</th>
      <th>Q29_B_Part_4</th>
      <th>Q29_B_Part_5</th>
      <th>Q29_B_Part_6</th>
      <th>Q29_B_Part_7</th>
      <th>Q29_B_Part_8</th>
      <th>Q29_B_Part_9</th>
      <th>Q29_B_Part_10</th>
      <th>Q29_B_Part_11</th>
      <th>Q29_B_Part_12</th>
      <th>Q29_B_Part_13</th>
      <th>Q29_B_Part_14</th>
      <th>Q29_B_Part_15</th>
      <th>Q29_B_Part_16</th>
      <th>Q29_B_Part_17</th>
      <th>Q29_B_OTHER</th>
      <th>Q31_B_Part_1</th>
      <th>Q31_B_Part_2</th>
      <th>Q31_B_Part_3</th>
      <th>Q31_B_Part_4</th>
      <th>Q31_B_Part_5</th>
      <th>Q31_B_Part_6</th>
      <th>Q31_B_Part_7</th>
      <th>Q31_B_Part_8</th>
      <th>Q31_B_Part_9</th>
      <th>Q31_B_Part_10</th>
      <th>Q31_B_Part_11</th>
      <th>Q31_B_Part_12</th>
      <th>Q31_B_Part_13</th>
      <th>Q31_B_Part_14</th>
      <th>Q31_B_OTHER</th>
      <th>Q33_B_Part_1</th>
      <th>Q33_B_Part_2</th>
      <th>Q33_B_Part_3</th>
      <th>Q33_B_Part_4</th>
      <th>Q33_B_Part_5</th>
      <th>Q33_B_Part_6</th>
      <th>Q33_B_Part_7</th>
      <th>Q33_B_OTHER</th>
      <th>Q34_B_Part_1</th>
      <th>Q34_B_Part_2</th>
      <th>Q34_B_Part_3</th>
      <th>Q34_B_Part_4</th>
      <th>Q34_B_Part_5</th>
      <th>Q34_B_Part_6</th>
      <th>Q34_B_Part_7</th>
      <th>Q34_B_Part_8</th>
      <th>Q34_B_Part_9</th>
      <th>Q34_B_Part_10</th>
      <th>Q34_B_Part_11</th>
      <th>Q34_B_OTHER</th>
      <th>Q35_B_Part_1</th>
      <th>Q35_B_Part_2</th>
      <th>Q35_B_Part_3</th>
      <th>Q35_B_Part_4</th>
      <th>Q35_B_Part_5</th>
      <th>Q35_B_Part_6</th>
      <th>Q35_B_Part_7</th>
      <th>Q35_B_Part_8</th>
      <th>Q35_B_Part_9</th>
      <th>Q35_B_Part_10</th>
      <th>Q35_B_OTHER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1838</td>
      <td>35-39</td>
      <td>Man</td>
      <td>Colombia</td>
      <td>Doctoral degree</td>
      <td>Student</td>
      <td>5-10 years</td>
      <td>Python</td>
      <td>R</td>
      <td>SQL</td>
      <td>C</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Javascript</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MATLAB</td>
      <td>NaN</td>
      <td>Other</td>
      <td>Python</td>
      <td>Jupyter (JupyterLab, Jupyter Notebooks, etc)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Visual Studio Code (VSCode)</td>
      <td>NaN</td>
      <td>Spyder</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kaggle Notebooks</td>
      <td>Colab Notebooks</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A cloud computing platform (AWS, Azure, GCP, h...</td>
      <td>GPUs</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 times</td>
      <td>Matplotlib</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Geoplotlib</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1-2 years</td>
      <td>NaN</td>
      <td>TensorFlow</td>
      <td>Keras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Xgboost</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Decision Trees or Random Forests</td>
      <td>Gradient Boosting Machines (xgboost, lightgbm,...</td>
      <td>Bayesian Approaches</td>
      <td>NaN</td>
      <td>Dense Neural Networks (MLPs, etc)</td>
      <td>Convolutional Neural Networks</td>
      <td>NaN</td>
      <td>Recurrent Neural Networks</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Image classification and other general purpose...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Word embeddings/vectors (GLoVe, fastText, word...</td>
      <td>NaN</td>
      <td>Contextualized embeddings (ELMo, CoVe)</td>
      <td>Transformer language models (GPT-3, BERT, XLne...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Coursera</td>
      <td>NaN</td>
      <td>Kaggle Learn Courses</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>University Courses (resulting in a university ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Basic statistical software (Microsoft Excel, G...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kaggle (notebooks, forums, etc)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Journal Publications (peer-reviewed journals, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Amazon Web Services (AWS)</td>
      <td>Microsoft Azure</td>
      <td>Google Cloud Platform (GCP)</td>
      <td>IBM Cloud / Red Hat</td>
      <td>NaN</td>
      <td>SAP Cloud</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Azure Cloud Services</td>
      <td>Microsoft Azure Container Instances</td>
      <td>Azure Functions</td>
      <td>Google Cloud Compute Engine</td>
      <td>Google Cloud Functions</td>
      <td>Google Cloud Run</td>
      <td>Google Cloud App Engine</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Amazon SageMaker</td>
      <td>Amazon Forecast</td>
      <td>Amazon Rekognition</td>
      <td>Azure Machine Learning Studio</td>
      <td>Azure Cognitive Services</td>
      <td>Google Cloud AI Platform / Google Cloud ML En...</td>
      <td>Google Cloud Video AI</td>
      <td>Google Cloud Natural Language</td>
      <td>Google Cloud Vision AI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MongoDB</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Microsoft SQL Server</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Google Cloud BigQuery</td>
      <td>Google Cloud SQL</td>
      <td>Google Cloud Firestore</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Microsoft Power BI</td>
      <td>Amazon QuickSight</td>
      <td>Google Data Studio</td>
      <td>NaN</td>
      <td>Tableau</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SAP Analytics Cloud</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Automated data augmentation (e.g. imgaug, albu...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Automated hyperparameter tuning (e.g. hyperopt...</td>
      <td>Automation of full ML pipelines (e.g. Google C...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Google Cloud AutoML</td>
      <td>NaN</td>
      <td>Databricks AutoML</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Auto-Keras</td>
      <td>Auto-Sklearn</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TensorBoard</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>289287</td>
      <td>30-34</td>
      <td>Man</td>
      <td>United States of America</td>
      <td>Master’s degree</td>
      <td>Data Engineer</td>
      <td>5-10 years</td>
      <td>Python</td>
      <td>R</td>
      <td>SQL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Python</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Visual Studio</td>
      <td>NaN</td>
      <td>PyCharm</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sublime Text</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Colab Notebooks</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A personal computer or laptop</td>
      <td>GPUs</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 times</td>
      <td>Matplotlib</td>
      <td>Seaborn</td>
      <td>NaN</td>
      <td>Ggplot / ggplot2</td>
      <td>Shiny</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1-2 years</td>
      <td>Scikit-learn</td>
      <td>TensorFlow</td>
      <td>Keras</td>
      <td>PyTorch</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Linear or Logistic Regression</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Convolutional Neural Networks</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Transformer Networks (BERT, gpt-3, etc)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Image segmentation methods (U-Net, Mask R-CNN,...</td>
      <td>NaN</td>
      <td>Image classification and other general purpose...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Contextualized embeddings (ELMo, CoVe)</td>
      <td>Transformer language models (GPT-3, BERT, XLne...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10,000 or more employees</td>
      <td>20+</td>
      <td>We have well established ML methods (i.e., mod...</td>
      <td>Analyze and understand data to influence produ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Do research that advances the state of the art...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100,000-124,999</td>
      <td>$100,000 or more ($USD)</td>
      <td>Amazon Web Services (AWS)</td>
      <td>Microsoft Azure</td>
      <td>Google Cloud Platform (GCP)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Amazon EC2</td>
      <td>AWS Lambda</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Azure Functions</td>
      <td>Google Cloud Compute Engine</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Amazon SageMaker</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PostgresSQL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Amazon Redshift</td>
      <td>Amazon Athena</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PostgresSQL</td>
      <td>Amazon QuickSight</td>
      <td>Microsoft Power BI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Tableau</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Microsoft Power BI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No / None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No / None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GitHub</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Coursera</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DataCamp</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Udemy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Business intelligence software (Salesforce, Ta...</td>
      <td>Twitter (data science influencers)</td>
      <td>NaN</td>
      <td>Reddit (r/machinelearning, etc)</td>
      <td>Kaggle (notebooks, forums, etc)</td>
      <td>Course Forums (forums.fast.ai, Coursera forums...</td>
      <td>YouTube (Kaggle YouTube, Cloud AI Adventures, ...</td>
      <td>NaN</td>
      <td>Blogs (Towards Data Science, Analytics Vidhya,...</td>
      <td>NaN</td>
      <td>Slack Communities (ods.ai, kagglenoobs, etc)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Responders in different countries**


```python
country_count = df_1['Q3'].value_counts().reset_index()
country_count
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
      <th>index</th>
      <th>Q3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>India</td>
      <td>5851</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States of America</td>
      <td>2237</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Other</td>
      <td>1388</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>694</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Japan</td>
      <td>638</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Russia</td>
      <td>582</td>
    </tr>
    <tr>
      <th>6</th>
      <td>United Kingdom of Great Britain and Northern I...</td>
      <td>489</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nigeria</td>
      <td>476</td>
    </tr>
    <tr>
      <th>8</th>
      <td>China</td>
      <td>474</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Germany</td>
      <td>404</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Turkey</td>
      <td>344</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Spain</td>
      <td>336</td>
    </tr>
    <tr>
      <th>12</th>
      <td>France</td>
      <td>330</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Canada</td>
      <td>301</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Indonesia</td>
      <td>290</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Pakistan</td>
      <td>283</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Taiwan</td>
      <td>267</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Italy</td>
      <td>267</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Australia</td>
      <td>231</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Mexico</td>
      <td>227</td>
    </tr>
    <tr>
      <th>20</th>
      <td>South Korea</td>
      <td>190</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Egypt</td>
      <td>179</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Colombia</td>
      <td>177</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Ukraine</td>
      <td>170</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Iran, Islamic Republic of...</td>
      <td>162</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Kenya</td>
      <td>153</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Netherlands</td>
      <td>151</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Singapore</td>
      <td>149</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Poland</td>
      <td>148</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Viet Nam</td>
      <td>147</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Bangladesh</td>
      <td>143</td>
    </tr>
    <tr>
      <th>31</th>
      <td>South Africa</td>
      <td>141</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Argentina</td>
      <td>134</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Morocco</td>
      <td>133</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Malaysia</td>
      <td>133</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Thailand</td>
      <td>132</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Portugal</td>
      <td>122</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Greece</td>
      <td>111</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Tunisia</td>
      <td>99</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Philippines</td>
      <td>99</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Israel</td>
      <td>97</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Peru</td>
      <td>95</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Chile</td>
      <td>85</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Sweden</td>
      <td>78</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Republic of Korea</td>
      <td>76</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Saudi Arabia</td>
      <td>76</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Sri Lanka</td>
      <td>72</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Switzerland</td>
      <td>68</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Nepal</td>
      <td>62</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Romania</td>
      <td>61</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Belgium</td>
      <td>60</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Belarus</td>
      <td>59</td>
    </tr>
    <tr>
      <th>52</th>
      <td>United Arab Emirates</td>
      <td>59</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Ireland</td>
      <td>54</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Ghana</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>




```python
country_count.columns = ['country', 'people']
country_count.columns
```



```
    Index(['country', 'people'], dtype='object')
```



```python
# To plot the data in the map , I will use plotly dataset to get the country code
country_code = pd.read_csv('../input/plotly-country-code-mapping/2014_world_gdp_with_codes.csv')
country_code.columns = [i.lower() for i in country_code.columns]

```


```python
country_count.loc[country_count['country'] == 'United States of America', 'country'] = 'United States'
country_count.loc[country_count['country'] == 'United Kingdom of Great Britain and Northern Ireland', 'country'] = 'United Kingdom'
country_count.loc[country_count['country'] == 'South Korea', 'country'] = '"Korea, South"'
country_count.loc[country_count['country'] == 'Viet Nam', 'country'] = 'Vietnam'
country_count.loc[country_count['country'] == 'Iran, Islamic Republic of...', 'country'] = 'Iran'
country_count.loc[country_count['country'] == 'Hong Kong (S.A.R.)', 'country'] = 'Hong Kong'
country_count.loc[country_count['country'] == 'Republic of Korea', 'country'] = '"Korea, North"'
country_count = pd.merge(country_count, country_code, on='country')
country_count
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
      <th>country</th>
      <th>people</th>
      <th>gdp (billions)</th>
      <th>code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>India</td>
      <td>5851</td>
      <td>2048.00</td>
      <td>IND</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>2237</td>
      <td>17420.00</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>694</td>
      <td>2244.00</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Japan</td>
      <td>638</td>
      <td>4770.00</td>
      <td>JPN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Russia</td>
      <td>582</td>
      <td>2057.00</td>
      <td>RUS</td>
    </tr>
    <tr>
      <th>5</th>
      <td>United Kingdom</td>
      <td>489</td>
      <td>2848.00</td>
      <td>GBR</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Nigeria</td>
      <td>476</td>
      <td>594.30</td>
      <td>NGA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>China</td>
      <td>474</td>
      <td>10360.00</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>404</td>
      <td>3820.00</td>
      <td>DEU</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Turkey</td>
      <td>344</td>
      <td>813.30</td>
      <td>TUR</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Spain</td>
      <td>336</td>
      <td>1400.00</td>
      <td>ESP</td>
    </tr>
    <tr>
      <th>11</th>
      <td>France</td>
      <td>330</td>
      <td>2902.00</td>
      <td>FRA</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Canada</td>
      <td>301</td>
      <td>1794.00</td>
      <td>CAN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Indonesia</td>
      <td>290</td>
      <td>856.10</td>
      <td>IDN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Pakistan</td>
      <td>283</td>
      <td>237.50</td>
      <td>PAK</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Taiwan</td>
      <td>267</td>
      <td>529.50</td>
      <td>TWN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Italy</td>
      <td>267</td>
      <td>2129.00</td>
      <td>ITA</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Australia</td>
      <td>231</td>
      <td>1483.00</td>
      <td>AUS</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Mexico</td>
      <td>227</td>
      <td>1296.00</td>
      <td>MEX</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Egypt</td>
      <td>179</td>
      <td>284.90</td>
      <td>EGY</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Colombia</td>
      <td>177</td>
      <td>400.10</td>
      <td>COL</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Ukraine</td>
      <td>170</td>
      <td>134.90</td>
      <td>UKR</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Iran</td>
      <td>162</td>
      <td>402.70</td>
      <td>IRN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Kenya</td>
      <td>153</td>
      <td>62.72</td>
      <td>KEN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Netherlands</td>
      <td>151</td>
      <td>880.40</td>
      <td>NLD</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Singapore</td>
      <td>149</td>
      <td>307.90</td>
      <td>SGP</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Poland</td>
      <td>148</td>
      <td>552.20</td>
      <td>POL</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Vietnam</td>
      <td>147</td>
      <td>187.80</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Bangladesh</td>
      <td>143</td>
      <td>186.60</td>
      <td>BGD</td>
    </tr>
    <tr>
      <th>29</th>
      <td>South Africa</td>
      <td>141</td>
      <td>341.20</td>
      <td>ZAF</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Argentina</td>
      <td>134</td>
      <td>536.20</td>
      <td>ARG</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Morocco</td>
      <td>133</td>
      <td>112.60</td>
      <td>MAR</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Malaysia</td>
      <td>133</td>
      <td>336.90</td>
      <td>MYS</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Thailand</td>
      <td>132</td>
      <td>373.80</td>
      <td>THA</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Portugal</td>
      <td>122</td>
      <td>228.20</td>
      <td>PRT</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Greece</td>
      <td>111</td>
      <td>246.40</td>
      <td>GRC</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Tunisia</td>
      <td>99</td>
      <td>49.12</td>
      <td>TUN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Philippines</td>
      <td>99</td>
      <td>284.60</td>
      <td>PHL</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Israel</td>
      <td>97</td>
      <td>305.00</td>
      <td>ISR</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Peru</td>
      <td>95</td>
      <td>208.20</td>
      <td>PER</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Chile</td>
      <td>85</td>
      <td>264.10</td>
      <td>CHL</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Sweden</td>
      <td>78</td>
      <td>559.10</td>
      <td>SWE</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Saudi Arabia</td>
      <td>76</td>
      <td>777.90</td>
      <td>SAU</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Sri Lanka</td>
      <td>72</td>
      <td>71.57</td>
      <td>LKA</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Switzerland</td>
      <td>68</td>
      <td>679.00</td>
      <td>CHE</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Nepal</td>
      <td>62</td>
      <td>19.64</td>
      <td>NPL</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Romania</td>
      <td>61</td>
      <td>199.00</td>
      <td>ROU</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Belgium</td>
      <td>60</td>
      <td>527.80</td>
      <td>BEL</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Belarus</td>
      <td>59</td>
      <td>75.25</td>
      <td>BLR</td>
    </tr>
    <tr>
      <th>49</th>
      <td>United Arab Emirates</td>
      <td>59</td>
      <td>416.40</td>
      <td>ARE</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Ireland</td>
      <td>54</td>
      <td>245.80</td>
      <td>IRL</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Ghana</td>
      <td>52</td>
      <td>35.48</td>
      <td>GHA</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = [ dict(
        type = 'choropleth',
        locations = country_count['code'],
        z = country_count['people'],
        text = country_count['country'],
        colorscale = 'Viridis',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Responders'),
      ) ]

layout = dict(
    title = 'Responders by country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )
```


<div>                            <div id="350bf5dc-19a9-41e6-9a8e-e7fbf2633351" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("350bf5dc-19a9-41e6-9a8e-e7fbf2633351")) {                    Plotly.newPlot(                        "350bf5dc-19a9-41e6-9a8e-e7fbf2633351",                        [{"autocolorscale": false, "colorbar": {"autotick": false, "title": "Responders"}, "colorscale": "Viridis", "locations": ["IND", "USA", "BRA", "JPN", "RUS", "GBR", "NGA", "CHN", "DEU", "TUR", "ESP", "FRA", "CAN", "IDN", "PAK", "TWN", "ITA", "AUS", "MEX", "EGY", "COL", "UKR", "IRN", "KEN", "NLD", "SGP", "POL", "VNM", "BGD", "ZAF", "ARG", "MAR", "MYS", "THA", "PRT", "GRC", "TUN", "PHL", "ISR", "PER", "CHL", "SWE", "SAU", "LKA", "CHE", "NPL", "ROU", "BEL", "BLR", "ARE", "IRL", "GHA"], "marker": {"line": {"color": "rgb(180,180,180)", "width": 0.5}}, "reversescale": true, "text": ["India", "United States", "Brazil", "Japan", "Russia", "United Kingdom", "Nigeria", "China", "Germany", "Turkey", "Spain", "France", "Canada", "Indonesia", "Pakistan", "Taiwan", "Italy", "Australia", "Mexico", "Egypt", "Colombia", "Ukraine", "Iran", "Kenya", "Netherlands", "Singapore", "Poland", "Vietnam", "Bangladesh", "South Africa", "Argentina", "Morocco", "Malaysia", "Thailand", "Portugal", "Greece", "Tunisia", "Philippines", "Israel", "Peru", "Chile", "Sweden", "Saudi Arabia", "Sri Lanka", "Switzerland", "Nepal", "Romania", "Belgium", "Belarus", "United Arab Emirates", "Ireland", "Ghana"], "type": "choropleth", "z": [5851, 2237, 694, 638, 582, 489, 476, 474, 404, 344, 336, 330, 301, 290, 283, 267, 267, 231, 227, 179, 177, 170, 162, 153, 151, 149, 148, 147, 143, 141, 134, 133, 133, 132, 122, 111, 99, 99, 97, 95, 85, 78, 76, 72, 68, 62, 61, 60, 59, 59, 54, 52]}],                        {"geo": {"projection": {"type": "Mercator"}, "showcoastlines": false, "showframe": false}, "title": "Responders by country"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('350bf5dc-19a9-41e6-9a8e-e7fbf2633351');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


This plot shows clearly that most respondent are from India, USA and Brazil. 

There is not a lot of aspiring Data scientist in Africa, as the Map show only few countries such as South Africa and Nigeria. 

Most country with less or not respondent are undeveloped than countries with most respondent


**Comparing responders by countries**

It would be difficult to analyse each and every country so i have decided to only take the top 3 countries (India, USA and Brazil ) separately and other countries will be grouped in one category.



```python
df_1['Q3_orig'] = df_1['Q3']
df_1.loc[df['Q3'].isin(['United States of America', 'India', 'Brazil']) == False,
              'Q3'] = 'Other countries'

```

**How long did it take to answer the survey**



**Gender and age**


```python
data = []
for i in df_1['Q1'].unique():
    trace = go.Bar(
        x=df_1.loc[df_1['Q1'] == i, 'Q2'].value_counts().sort_index().index,
        y=df_1.loc[df_1['Q1'] == i, 'Q2'].value_counts().sort_index().values,
        name=i
    )
    data.append(trace)
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')
```

