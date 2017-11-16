
# coding: utf-8

# In[83]:


from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[64]:


df = pd.read_csv("val_traditional.csv")


# In[65]:


df.head(20)


# In[66]:


val_sorted = df.sort_values(by='traditional')
val_sorted.describe()


# In[67]:


from scipy.spatial.distance import pdist, squareform


# In[68]:


meals_only = pd.DataFrame(df['traditional'])


# In[69]:


# dists = pdist(meals_only, fuzz.token_sort_ratio)

import pickle as pkl
with open('val_dists_try1.pkl', 'rb') as f:
    dists = pkl.load(f)


# In[70]:


dists


# In[71]:


mat = squareform(dists)


# In[72]:


##CREDIT TO erikm0111

def sumRow(matrix, i):
    return np.sum(matrix[i,:])
 
def determineRow(matrix):
    maxNumOfOnes = -1
    row = -1
    for i in range(len(matrix)):
        if maxNumOfOnes < sumRow(matrix, i):
            maxNumOfOnes = sumRow(matrix, i)
            row = i
    return row
 
def addIntoGroup(matrix, ind):
    change = True
    indexes = []
    for col in range(len(matrix)):
        if matrix[ind, col] == 1:
            indexes.append(col)
    while change == True:
        change = False
        numIndexes = len(indexes)
        for i in indexes:
            for col in range(len(matrix)):
                if matrix[i, col] == 1:
                    if col not in indexes:
                        indexes.append(col)
        numIndexes2 = len(indexes)
        if numIndexes != numIndexes2:
            change = True
    return indexes
 
def deleteChosenRowsAndCols(matrix, indexes):
    for i in indexes:
        matrix[i,:] = 0
        matrix[:,i] = 0
    return matrix
def categorizeIntoClusters(matrix):
    groups = []
    while np.sum(matrix) > 0:
        group = []
        row = determineRow(matrix)
        indexes = addIntoGroup(matrix, row)
        groups.append(indexes)
        matrix = deleteChosenRowsAndCols(matrix, indexes)
    return groups


# In[73]:


len(mat)


# In[74]:


def buildThreshholdMatrix(prev, threshold):
	numOfSamples = len(prev)
	matrix = np.zeros(shape=(numOfSamples, numOfSamples))
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			dist = prev[i,j]
			if dist > threshold:
				matrix[i,j] = 1
	return matrix


# In[75]:


thresh = buildThreshholdMatrix(mat, 90)


# In[76]:


thresh


# In[77]:


groups = categorizeIntoClusters(thresh)


# In[78]:


len(groups)


# In[87]:


import seaborn as sns
sns.distplot(dists)


# In[88]:


plt.show()


# In[89]:


df['clusters'] = 0
for i in range(len(groups)):
    for j in range(len(groups[i])):
        df.loc[groups[i][j],'clusters'] = i


# In[90]:


df[df['clusters'] == 216]


# In[91]:


calzone = df[df.traditional.str.contains('Calzone', na = False)]


# In[92]:


calzone.head(20)


# In[93]:


print(calzone.loc[3359,'traditional'])


# In[94]:


print(calzone.loc[3253,'traditional'])


# In[95]:


fuzz.token_set_ratio(calzone.loc[3359,'traditional'],calzone.loc[3253,'traditional'])


# In[96]:


import utils


# In[97]:


def _token_print_set(s1, s2, partial=True, force_ascii=True, full_process=True):
    """Find all alphanumeric tokens in each string...
        - treat them as a set
        - construct two strings of the form:
            <sorted_intersection><sorted_remainder>
        - take ratios of those two strings
        - controls for unordered partial matches"""

    p1 = utils.full_process(s1, force_ascii=force_ascii) if full_process else s1
    p2 = utils.full_process(s2, force_ascii=force_ascii) if full_process else s2

    if not utils.validate_string(p1):
        return 0
    if not utils.validate_string(p2):
        return 0

    # pull tokens
    tokens1 = set(p1.split())
    tokens2 = set(p2.split())

    intersection = tokens1.intersection(tokens2)
    diff1to2 = tokens1.difference(tokens2)
    diff2to1 = tokens2.difference(tokens1)
    
    print(intersection)
    print(diff1to2)
    print(diff2to1)

    sorted_sect = " ".join(sorted(intersection))
    sorted_1to2 = " ".join(sorted(diff1to2))
    sorted_2to1 = " ".join(sorted(diff2to1))

    combined_1to2 = sorted_sect + " " + sorted_1to2
    combined_2to1 = sorted_sect + " " + sorted_2to1

    # strip
    sorted_sect = sorted_sect.strip()
    combined_1to2 = combined_1to2.strip()
    combined_2to1 = combined_2to1.strip()


# In[98]:


_token_print_set(calzone.loc[3359,'traditional'],calzone.loc[3253,'traditional'])


# In[99]:


common = pd.Series(' '.join(calzone['traditional']).lower().split()).value_counts()


# In[100]:


common
counts = common.value_counts()


# In[101]:


sorted= counts.sort_index()


# In[102]:


index = sorted.iloc[-1]


# In[103]:


sorted.index[-1]


# In[104]:


common[common == 13].index.values


# In[105]:


def getTop(series, min):
    common = series.value_counts()
    counts = common.value_counts()
    sorted = counts.sort_index()
    num = 0
    counts = []
    end = 1
    while num < min:
        num += sorted.iloc[-end]
        counts.append(sorted.index[-end])
        end+=1
    label = ' '
    print(counts)
    for count in counts:
        print(count)
        label = label.join(common[common == count].index.values)
    return label


# In[106]:


getTop(pd.Series(' '.join(calzone['traditional']).lower().split()), 4)


# In[107]:


df['abbrev_name'] = ''


# In[108]:


df_copy = df


# In[110]:


df.head()


# In[109]:


groups = df_copy.groupby('clusters2')


# In[ ]:


groups.groups


# In[ ]:


for group in groups:
    df[group]


# In[290]:


df_groups = df_copy.join(df_copy.groupby('clusters2').getTop())


# In[227]:


' '.join(calzone['traditional'])


# In[111]:


thresh2 = buildThreshholdMatrix(mat,40)


# In[112]:


groups2 = categorizeIntoClusters(thresh2)


# In[113]:


df['clusters2'] = 0
for i in range(len(groups2)):
    for j in range(len(groups2[i])):
        df.loc[groups2[i][j],'clusters2'] = i


# In[124]:


mat.shape
rows = mat.shape[0]
groups = dict.fromkeys(range(0,row-1))
for i in range(0, row-1):
    for(j in range(i,row -1)):
        if(mat[])


# In[126]:


mat.itemsize


# In[44]:


import matplotlib.pyplot as plt
 
labels = range(0,mat.size)

fig, ax = plt.subplots(figsize=(20,20))
cax = ax.matshow(mat, interpolation='nearest')
ax.grid(True)
plt.title('Val Data Similarity matrix')
plt.xticks(range(33), labels, rotation=90);
plt.yticks(range(33), labels);
fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75,.8,.85,.90,.95,1])
plt.show()


# In[21]:


import pickle
pickle.dump(dists, open("val_dists_try1.pkl", "wb"))


# In[22]:


from sklearn.cluster import DBSCAN


# In[67]:


db = DBSCAN(eps = 0.1, min_samples = 0, metric = "precomputed").fit_predict(normalized)
db


# In[63]:


normalized = np.divide(mat, 100)
normalized


# In[68]:


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


# In[77]:


link = linkage(dists)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    link,
    truncate_mode='lastp',
    p = 200,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[116]:


from scipy.cluster.hierarchy import fcluster
max_d = 200
clusters = fcluster(link, max_d, criterion='maxclust')
print(max(clusters))
clusters


# In[120]:


df['clusters'] = clusters
df[df['clusters'] == 1]


# In[41]:


import numpy as np
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


# In[19]:


squareform(dists)


# In[5]:


group = 1
grouping = dict.fromkeys(range(0,3525))
grouping[0] == None


# In[6]:


for index, row in df.iterrows():
    ratios = {}
    to_match = row['traditional']
    print('matching {index}'.format(index = index))
    for index, row in df.iterrows():
        ratio = fuzz.token_sort_ratio(to_match, index)
        ratios[index] = ratio
    grouping[index] = ratios

