def run_model_pipeline_grid(features,target):
    # Runs a pipeline for the model, in this case I've hardcoded it for a random forest model.
    from sklearn.cross_validation import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import RandomizedSearchCV    
    #Train test split is performed to get a better understanding of how the model is performing. 
    #It's important to look at to decide hyperparameters how the model will be generalized and behave when given new data.

    X_train, X_test, y_train, y_test = train_test_split(features,target,test_size = 0.3)
    clf = Pipeline([('clf', RandomForestClassifier())])
    clf.fit(X_train, y_train)  
    parameters = {'clf__n_estimators': (30,40,100,500),
                'clf__max_depth':(3,4,5)}
    ## Using a gridisearch to help tune the model parameters
    rs_clf= RandomizedSearchCV(clf, parameters, n_jobs=1)
    rs_clf= rs_clf.fit(X_train,y_train)
    print(X_train.shape,y_train.shape)
    
    print(rs_clf.best_score_) 
    rs_clf.cv_results_
    
    return(rs_clf.best_estimator_, X_train, X_test, y_train, y_test, features, target)
