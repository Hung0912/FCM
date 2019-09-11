import pandas as pd
import numpy as np
import random
import operator
import math

csv_data = pd.read_csv("SPECTF_New.csv")
# print(csv_data)
columns = list(csv_data.columns)
# print(columns)
class_labels = list(csv_data[columns[-1]])
# print(class_labels)
feartures = columns[:len(columns)-1]
# print(feartures)
data = csv_data[feartures]
# print(data)

# Number of Attributes
num_attr = len(columns) - 1

# Number of Clusters
k = 2

# Maximum number of iterations
max_iter = 100

# Number of data points
n = len(data)

# Fuzzy parameter
m = 2.00

# print(num_attr, k, max_iter, n, m)

def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat

def calculateClusterCenter(membership_mat):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = list()
    for j in range(k):
        x = cluster_mem_val[j]
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(data.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(n):
        x = list(data.iloc[i])
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat

def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

def FCM():
    membership_mat = initializeMembershipMatrix()
    curr = 0
    while curr <= max_iter:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        curr += 1
    return cluster_labels, cluster_centers

def accuracy(cluster_labels, class_labels):
    county = [0,0]
    countn = [0,0]
    tp = [0, 0]
    tn = [0, 0]
    fp = [0, 0]
    fn = [0, 0]
    
    for i in range(n):
        # Yes = 1, No = 0
        if cluster_labels[i] == 1 and class_labels[i] == 'Yes':
            tp[0] = tp[0] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'No':
            tn[0] = tn[0] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'No':
            fp[0] = fp[0] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'Yes':
            fn[0] = fn[0] + 1
    
    for i in range(n):
        # Yes = 0, No = 1
        if cluster_labels[i] == 0 and class_labels[i] == 'Yes':
            tp[1] = tp[1] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'No':
            tn[1] = tn[1] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'No':
            fp[1] = fp[1] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'Yes':
            fn[1] = fn[1] + 1
    
    a0 = float((tp[0] + tn[0]))/(tp[0] + tn[0] + fn[0] + fp[0])
    a1 = float((tp[1] + tn[1]))/(tp[1] + tn[1] + fn[1] + fp[1])
    p0 = float(tp[0])/(tp[0] + fp[0])
    p1 = float(tp[1])/(tp[1] + fp[1])
    r0 = float(tp[0])/(tp[0] + fn[0])
    r1 = float(tp[1])/(tp[1] + fn[1])
    
    accuracy = [a0*100,a1*100]
    precision = [p0*100,p1*100]
    recall = [r0*100,r1*100]
    
    return accuracy, precision, recall

if __name__ == "__main__":
    # 
    labels, centers = FCM()
    accuracy, precision, recall = accuracy(labels, class_labels)
    # print(class_labels)
    # print(labels)
    # print("Accuracy = " + str(accuracy))
    # print("Precision = " + str(precision))
    # print("Recall = " + str(recall))
    print("-----------CLUSTER 1 center-------------", centers[0])
    print("-----------CLUSTER 2 center-------------", centers[1])



