# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:48:42 2021

@author: Frank Vasquez
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, KFold
from sklearn.tree import export_graphviz

from io import StringIO
from IPython.display import Image
import pydotplus

# will be combined dataframe
def build_model(df, depth):
    print("Building model\n", "-"*20)
    # add features
    feature_cols = ['region_factor', 'Economy (GDP per Capita)', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity']
    pred_col = 'buckets'
    # use if dropping columns for predictive purposes
    subset_cols = feature_cols.copy()
    subset_cols.append(pred_col)
    df = df[subset_cols]
    
    # set up features
    x = df.loc[:,feature_cols].values
    label_values = list(df[pred_col].unique())
    y = df.loc[:,pred_col].values
    
    # randomly split data for model training and testing
    # default size splits 75/25
    # setting random state will allow for reproducible results
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
    
    # building model
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train, y_train)
    
    # extract feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance.nlargest().plot(kind='barh')
    plt.savefig('feature_importance.png', dpi=500, bbox_inches='tight')
    
    # visualization
    tree.plot_tree(model)
    
    # complex, pretty visualization
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=label_values)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    graph.write_png("dtree.png")
    
    # prediction
    y_pred = model.predict(x_test)
    
    # evaluate performance
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plot_confusion_matrix(model, x_test, y_test)
    plt.savefig('confusion_matrix.png', dpi=500)
    print("Accuracy: ", accuracy_score(y_test,y_pred))
    print(classification_report(y_test, y_pred))
    
    return model