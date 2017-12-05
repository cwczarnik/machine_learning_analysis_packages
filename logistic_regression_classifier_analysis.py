from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve

def logistic_regression_analysis(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    prec, rec, thresh_ = precision_recall_curve(y_test,y_pred)
    fpr,tpr, thresh = roc_curve(y_test,y_pred)
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.plot(rec,prec)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.subplot(1,2,2)
    plt.plot(fpr,tpr)
    plt.plot([1,0], [1,0], 'k--', lw = 2)
    plt.xlabel('fpr')
    plt.ylabel('tpr')

    plt.show()
    # F1 = 2 * (prec * rec) / (prec + rec)
    thresh = list(thresh)
    # thresh.append(1)
    plt.plot(thresh,tpr)
    plt.title('TPR Versus Threshold')
    plt.ylabel('tpr')
    plt.xlabel('Threshold')
    plt.show()

    plt.show()
    # F1 = 2 * (prec * rec) / (prec + rec)
    thresh = list(thresh)
    # thresh.append(1)
    plt.plot(thresh,fpr)
    plt.title('FPR Versus Threshold')
    plt.ylabel('fpr')
    plt.xlabel('Threshold')
    plt.show()

    odds = np.exp(model.coef_[0])*np.sign(model.coef_[0])
    sorted_index = odds.argsort()
    fig, ax = plt.subplots(figsize=(6, 11))  
    width = 0.75 # the width of the bars 
    ind = np.arange(X_test.shape[1])  # the x locations for the groups
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(X_test.columns[sorted_index])
    ax.barh(ind, odds[sorted_index])
    plt.title('Odds Ratio w/ sign for each feature')
    plt.show()

    print("At threshold = 0.5")
    # It is worse to class a customer as good when they are bad, 
    # than it is to class a customer as bad when they are good.
    print(metrics.classification_report(y_test,y_pred > 0.5))
    print('accuracy: ',metrics.accuracy_score(y_test,y_pred > 0.5))
    return(model)