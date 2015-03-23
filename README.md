# RandomForest
Random Forest Implementation on Spark
Random forests have a avour of bagging along with randomness. In a typical decision tree, the nodes are split using the best 
split among all variables. However, in the case of Random forests, each node is split using a subset of the total features 
available. Although it sounds absurd, turns out better in terms of accuracy as compared to many other classifiers. 
It is easier to understand in the sense that it requires only two parameters : the number of features to be used to create each 
subset of data and the number of trees in the forest.

PMML uses XML to dene the data mining models(could be any model from Linear Regression, Decision Trees, Association rules, 
Naive Bayes, etc.. the list is pretty exhaustive). The structure of the XML is governed by an XML schema. In brief, a PMML 
document is an XML document with root element as PMML.
Spark currently does not have any implementation to convert its machine learning models into PMML. 
We have implemented a framework to convert Random Forest model trained in Apache Spark into PMML

This Random Forest implementation makes use of the Apache Spark RDD's to store the chunk of data in memory which makes Spark so efficient
in data processing.
Apache Spark has has an existing implementation of Decision Trees and Regression Trees which we have made use of, to build our 
forests on. We have also implemented a utility program which converts 
the Spark trained Random Forest model into PMML[6] which is a standard for converting machine learning models to XML format.
