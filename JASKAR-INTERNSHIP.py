#!/usr/bin/env python
# coding: utf-8

# # #import libraries

# In[75]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_score, recall_score, precision_recall_curve
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# # loading and read the data

# In[7]:


import pandas as pd
df=pd.read_csv("Datasets.csv")
df


# # check the missing

# In[9]:


df.isnull().sum()


# In[8]:


df.head()


# In[11]:


df.columns


# In[13]:


df.describe


# In[28]:


df.info()


# # ploting

# In[16]:


df.plot(y=['Packets_Sent',
       'Packets_Received', 'SRPR', 'class'], kind="bar")


# In[24]:


df.plot(y=['Packets_Sent',
       'Packets_Received'], kind="hist")


# In[35]:


plt.figure(figsize=(22, 6))

# Passenger Count
plt.subplot(121)
sns.countplot(df['AvgDuration'])
plt.xlabel('Sender_IP')
plt.ylabel('Target_IP')

# vendor_id
plt.subplot(122)
sns.countplot(df['Duration'])
plt.xlabel('Sender_Port')
plt.ylabel('Target_Port	')


# # train and test the data

# In[58]:


df_baseline = ['Sender_IP', 'Sender_Port', 'Target_IP', 'Target_Port',
       'Transport_Protocol', 'Duration']


# In[59]:


df_baseline


# In[ ]:


# Splitting the data into Train and Validation set
xtrain, xtest, ytrain, ytest = train_test_split(df_baseline,y_all,test_size=1/3, random_state=11, stratify = y_all)


# In[ ]:


model = LogisticRegression()
model.fit(xtrain,ytrain)
pred = model.predict_proba(xtest)[:,1]


# In[ ]:


#boxplot


# In[79]:


import pandas as pd
df=pd.read_csv("Datasets.csv")
df.head()


# In[83]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[84]:


sns.boxplot(df['Duration'])
plt.show()


# In[87]:


# importing the modules 
import numpy as np 
import seaborn as sn 
import matplotlib.pyplot as plt 

# generating 2-D 10x10 matrix of random numbers 
# from 1 to 100 
data = np.random.randint(low = 1, 
						high = 100, 
						size = (10, 10)) 
print("The data to be plotted:\n") 
print(data) 

# plotting the heatmap 
hm = sn.heatmap(data = data) 

# displaying the plotted heatmap 
plt.show()


# # prediction of heat map

# In[90]:


plt.figure(figsize=(12, 6))
df = df.drop(['Sender_IP', 'Sender_Port', 'Target_IP', 'Target_Port',
       'Transport_Protocol'],
        axis=1)
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                 linewidths=.2, cmap="YlGnBu")


# # USES

# In[ ]:


1.Enhancing IoT Network Performance:

The hybrid deep learning approach can be applied to IoT networks to identify bottlenecks and
optimize network performance. This is crucial for maintaining low latency and high throughput in IoT applications.

2.Resource Management:

By detecting bottlenecks in IoT devices and networks, the hybrid deep learning model aids in 
efficient resource management. It allows for better utilization of computational resources, memory, and bandwidth.

3.Real-time Decision Making:

The ability to detect bottlenecks in real-time enables prompt decision-making. 
This is especially valuable in IoT scenarios where timely responses to data are essential, such as in industrial automation 
or healthcare monitoring.

4.Scalability of IoT Systems:

As IoT systems grow in complexity and scale, the hybrid deep learning approach ensures that the 
network can adapt to increased demands by proactively identifying and addressing bottlenecks.

5.Energy Efficiency:

Optimizing for bottlenecks contributes to energy-efficient IoT devices. By minimizing unnecessary 
computations or data transmissions, the hybrid deep learning model helps extend the battery life of edge devices 
in IoT networks.

6.Fault Tolerance:

Bottleneck detection aids in identifying potential points of failure or congestion in an IoT network. 
This information can be used to implement redundancy or rerouting strategies, improving the overall fault tolerance 
of the system.


# # conclusion:

# In[ ]:


1.Comprehensive Bottleneck Detection:

The hybrid deep learning approach combines the strengths of different models or techniques, 
providing a comprehensive solution for bottleneck detection. This may involve using traditional
machine learning algorithms alongside deep neural networks.

2.Adaptability to IoT Dynamics:

IoT environments are dynamic and subject to changes in data patterns and network conditions. 
The hybrid approach allows for adaptability to these dynamics, ensuring the model remains effective over time.

3.Cross-Layer Analysis:

The hybrid approach may enable cross-layer analysis, considering interactions between different layers 
of the IoT architecture. This holistic view enhances the models ability to identify complex bottlenecks 
that span multiple components.

4.Practical Deployment:

The effectiveness of the hybrid deep learning approach in bottleneck detection makes it a practical choice for
deployment in real-world IoT scenarios. It bridges the gap between theoretical models and practical applicability.

5.Continuous Improvement:

Ongoing monitoring and feedback from IoT deployments can be used to refine and improve the hybrid deep learning model. 
This ensures that it stays effective in addressing new challenges and evolving IoT landscapes.

6.Contributions to IoT Ecosystem:

By providing a reliable solution for bottleneck detection, the hybrid deep learning approach contributes to the overall
efficiency, reliability, and sustainability of the IoT ecosystem.


In conclusion, a hybrid deep learning approach for bottleneck detection in IoT has versatile uses,
ranging from improving network performance to enhancing resource management and scalability. Its adaptability, 
comprehensive analysis, and practical deployment make it a valuable tool for optimizing IoT systems in various domains.


# # SO THIS IS THE FINAL PREDICTION OF BOTTLENECK PREDICTION PROJECT .
