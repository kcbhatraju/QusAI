import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.nn import top_k
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, \
    roc_auc_score, balanced_accuracy_score, roc_curve, f1_score, precision_recall_curve, average_precision_score


def evaluate_acc_thresholds(model, x_test, y_test):
    print(model, '\n')
    y_pred_prob = model.predict(x_test)
    accuracy_scores = []
    thresholds = np.arange(0.01,0.99,0.01)
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_test[:,0],[1 if m > thresh else 0 for m in y_pred_prob[:,0]]))

    accuracies = np.array(accuracy_scores)
    max_accuracy_threshold =  thresholds[accuracies.argmax()]
    print('best thresh:',max_accuracy_threshold)

    y_pred_x = (model.predict(x_test)[:, 0] > max_accuracy_threshold).astype('float')
    print(confusion_matrix(y_test[:, 0], y_pred_x))
    print(classification_report(y_test[:, 0], y_pred_x))
    print("#############################")

def evaluate_ss_thresholds(model, x_test, y_test):
    print(model, '\n')
    y_pred_prob = model.predict(x_test)

    sensitivity_scores = []
    specificity_scores = []
    thresholds = np.arange(0.2,0.8,0.01)
    for thresh in thresholds:
        sensitivity_scores.append(recall_score(y_test[:,0],[1 if m >= thresh else 0 for m in y_pred_prob[:,0]]))
        specificity_scores.append(recall_score(y_test[:,0],[1 if m < thresh else 0 for m in y_pred_prob[:,0]]))
    
    plt.figure()
    plt.plot(sensitivity_scores,thresholds)
    plt.plot(specificity_scores,thresholds)
    plt.show

def evaluateModel(model, x_test, y_test, ax1, ax2, thresh=0.49):
    print('###########################')
    print('###########################')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss:', score[0])
    print('Accuracy:', score[1])

    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob[:,0]>thresh)
    print('thresh:',thresh)
    print(confusion_matrix(y_test[:,0], y_pred))
    print(classification_report(y_test[:,0], y_pred))
    print('Balanced ACC:', balanced_accuracy_score(y_test[:,0], y_pred))
    print('AUC:',roc_auc_score(y_test[:,0], y_pred_prob[:,0]))
    print('pAUC:',roc_auc_score(y_test[:,0], y_pred_prob[:,0], max_fpr = 0.50))
    print('microAUC:',roc_auc_score(y_test[:,0], y_pred_prob[:,0], average = 'micro'))
    print('macroAUC:',roc_auc_score(y_test[:,0], y_pred_prob[:,0], average = 'macro'))
    print('wAUC:',roc_auc_score(y_test[:,0], y_pred_prob[:,0], average = 'weighted'))
    print('sAUC:',roc_auc_score(y_test[:,0], y_pred_prob[:,0], average = 'samples'))
    
    evaluate_acc_thresholds(model, x_test, y_test)
    evaluate_ss_thresholds(model, x_test, y_test)
    
    fpr, tpr, _ = roc_curve(y_test[:,0], y_pred_prob[:,0])
    ax1.plot(fpr, tpr)
    ax1.set_title('Receiver Operator Curve')
    
    precision, recall, _ = precision_recall_curve(y_test[:,0], y_pred_prob[:,0])
    ax2.plot(recall, precision)
    ax2.set_title('Precision-Recall Curve')

    roc_auc = roc_auc_score(y_test[:,0], y_pred_prob[:,0])
    f101 = f1_score(y_test[:,0],(y_pred_prob[:,0]>thresh), average='binary')
    f102 = f1_score(y_test[:,1],(y_pred_prob[:,0]<thresh), average='binary')
    conf_mat = confusion_matrix(y_test[:,0], y_pred)
    
    return roc_auc, f101, f102, conf_mat

def auc_metric(true, pred):
    #We want strictly 1D arrays - cannot have (batch, 1), for instance
    true = K.flatten(true)
    pred = K.flatten(pred)

    #total number of elements in this batch
    total_count = K.shape(true)[0]

    #sorting the prediction values in descending order
    _, indices = top_k(pred, k = total_count)   
    #sorting the ground truth values based on the predictions above         
    sortedTrue = K.gather(true, indices)

    #getting the ground negative elements (already sorted above)
    negatives = 1 - sortedTrue

    #the true positive count per threshold
    tp_curve = K.cumsum(sortedTrue)

    #area under the curve
    auc = K.sum(tp_curve * negatives)

    #normalizing the result between 0 and 1
    total_count = K.cast(total_count, K.floatx())
    positive_count = K.sum(true)
    negative_count = total_count - positive_count
    total_area = positive_count * negative_count
    return  auc / total_area

def true_false_p_n(y_true, y_predict):
    t_positives = 0
    f_negatives = 0
    f_positives = 0
    for i in range(0, len(y_true)):
        if y_true[i] == 0:   
            if y_predict[i] == 0: #positive              
                t_positives = t_positives +1              
            elif y_predict[i] == 1:              
                f_negatives = f_negatives + 1              
        elif y_true[i] == 1:            
            if y_predict[i] == 0:               
                f_positives = f_positives +1   
    return t_positives, f_positives, f_negatives

def f1_compute(y_true, y_predict): 
    t_positives, f_positives, f_negatives = true_false_p_n(y_true, y_predict)
    precision = t_positives/(t_positives + f_positives)
    recall = t_positives/(t_positives + f_negatives)
    f1 = 2*(precision * recall)/(precision + recall)   
    return f1

def metric_asses_inv(model,x_test, y_test, name, thresh, ax1, ax2):
    print('###########################')
    print(name)
    print('###########################')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss:', score[0])
    print('Accuracy:', score[1])
    
    pred = model.predict(x_test)
    roc_auc = roc_auc_score(y_test[:,0], pred[:,0])
    # roc_auc2 = auc_metric(y_test[:,1], pred[:,0])

    pred_a =(pred[:,0]>thresh)
        
    f1_3 = f1_compute(y_test[:,0], pred_a)
    f1 = f1_score(y_test[:,0], pred_a, average='binary')
    conf_mat = confusion_matrix(y_test[:,0], pred_a)
    
    fpr, tpr,_ = roc_curve(y_test[:,0], pred[:,0])
    ax1.plot(fpr, tpr)
    ax1.set_title('Receiver Operator Curve')
    
    precision, recall, _ = precision_recall_curve(y_test[:,0], pred[:,0])
    ax2.plot(recall, precision)
    ax2.set_title('Precision-Recall Curve')

    return roc_auc, f1_3, f1, conf_mat

def metric_asses_inv_v2(model, x_test, y_test, name, thresh, ax1, ax2):
    print('###########################')
    print(name)
    print('###########################')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss:', score[0])
    print('Accuracy:', score[1])
    
    pred = model.predict(x_test)
    roc_auc = roc_auc_score(y_test[:,0], pred[:,0])
    roc_auc2 = auc_metric(y_test[:,1], pred[:,0])
    print('auc = %.3f' % roc_auc)
    print('auc2 = %.3f' % roc_auc2)

    pred_a =(pred[:,0]>thresh)
    
    print('>>>>Thresh = ')
    print(thresh)
    
    f1_3 = f1_compute(y_test[:,0], pred_a)
    print('F1 = %.3f' % f1_3)
    f1 = f1_score(y_test[:,0], pred_a, average='binary')
    print('F1 = %.3f' % f1)
    conf_mat = confusion_matrix(y_test[:,0], pred_a)
    print('Confusion Matrix')
    print(conf_mat)
    cm1 = conf_mat
    total1=sum(sum(cm1))
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy: ', accuracy1)

    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity: ', sensitivity1 )

    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Specificity: ', specificity1)

    average_precision = average_precision_score(y_test[:,0], pred[:,0])
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    
    fpr, tpr,_ = roc_curve(y_test[:,0], pred[:,0])
    ax1.plot(fpr, tpr)
    ax1.set_title('Receiver Operator Curve')
    
    precision, recall, _ = precision_recall_curve(y_test[:,0], pred[:,0])
    ax2.plot(recall, precision)
    ax2.set_title('Precision-Recall Curve')

    return roc_auc, f1_3, f1, conf_mat

def metric_asses(model, x_test, y_test, name, thresh):
    print('###########################')
    print(name)
    print('###########################')
    # Evaluation Metrics
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss:', score[0])
    print('Accuracy:', score[1])

    pred = model.predict(x_test)
    roc_auc = roc_auc_score(y_test[:,0], pred[:,0])
    roc_auc2 = auc_metric(y_test[:,1], pred[:,0])
    print('auc = %.3f' % roc_auc)
    print('auc2 = %.3f' % roc_auc2)
    
    pred_b =(pred[:,0]<thresh)
    
    print('>>>>Thresh = ')
    print(thresh)
    
    f1_3 = f1_compute(y_test[:,0], pred_b)
    print('F1 = %.3f' % f1_3)
    f1 = f1_score(y_test[:,0], pred_b, average='binary')
    print('F1 = %.3f' % f1)
    conf_mat = confusion_matrix(y_test[:,0], pred_b)
    print('Confusion Matrix')
    print(conf_mat)

    fpr, tpr,_ = roc_curve(y_test[:,0], pred[:,0])
    plt.plot(fpr, tpr)

    return roc_auc, f1_3, f1, conf_mat

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))