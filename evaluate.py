import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
    
def evaluate(result_dict, methods, query_is_labeled, folds=5):
    kl_mean = []
    kl_ci =[]

    max_skew_mean =[]
    max_skew_ci =[]

    #
    auc_mean =[]
    auc_ci =[]
    gap_mean =[]
    gap_ci =[]
    #
    classification_accuracy_mean =[]
    classification_accuracy_ci =[]
    retrieval_acc_mean =[]
    retrieval_acc_ci =[]
    total_acc_mean =[]
    total_acc_ci =[]
    #

    for m in methods:
        kl_avg_list = []
        max_skew_avg_list = []

        #
        auc_avg_list = []
        gap_avg_list = []
        #
        classification_accuracy_avg_list = []
        retrieval_accuracy_avg_list = []
        total_accuracy_avg_list = []

        for fold in range(folds):
            kl_list = []
            max_skew_list =[]

            #
            auc_list =[]
            gap_list =[]
            #
            classification_accuracy_list = []
            retrieval_accuracy_list = []
            total_accuracy_list = []

            for att in result_dict.keys():
                kl_list.append(result_dict[att][m][f"fold_{fold}"]['kl_prior'])
                max_skew_list.append(result_dict[att][m][f"fold_{fold}"]['max_skew_prior'])

                if query_is_labeled:
                    auc_list.append(result_dict[att][m][f"fold_{fold}"]['worst_auc_roc_val'])
                    gap_list.append(result_dict[att][m][f"fold_{fold}"]['auc_roc_gap'])
                    #
                    classification_accuracy_list.append(result_dict[att][m][f"fold_{fold}"]['classification_accuracy'])
                    retrieval_accuracy_list.append(result_dict[att][m][f"fold_{fold}"]['retrieval_accuracy'])
                    total_accuracy_list.append(result_dict[att][m][f"fold_{fold}"]['total_accuracy'])

            kl_avg_list.append(np.mean(kl_list))
            max_skew_avg_list.append(np.mean(max_skew_list))

            if query_is_labeled:
                auc_avg_list.append(np.mean(auc_list))
                gap_avg_list.append(np.mean(gap_list))
                #
                classification_accuracy_avg_list.append(np.mean(classification_accuracy_list))
                retr_acc_mean = {key: np.mean([d[key] for d in retrieval_accuracy_list]) for key in retrieval_accuracy_list[0]}
                retrieval_accuracy_avg_list.append(retr_acc_mean)
                total_accuracy_avg_list.append(np.mean(total_accuracy_list))

        _m, _h = mean_confidence_interval(kl_avg_list)
        kl_mean.append(_m)
        kl_ci.append(_h)

        _m, _h = mean_confidence_interval(max_skew_avg_list)
        max_skew_mean.append(_m)
        max_skew_ci.append(_h)

        if query_is_labeled:
            _m, _h = mean_confidence_interval(auc_avg_list)
            auc_mean.append(_m)
            auc_ci.append(_h)
            _m, _h = mean_confidence_interval(gap_avg_list)
            gap_mean.append(_m)
            gap_ci.append(_h)
            #
            _m, _h = mean_confidence_interval(classification_accuracy_avg_list)
            classification_accuracy_mean.append(_m)
            classification_accuracy_ci.append(_h)

            m = {}
            h = {}
            print(retrieval_accuracy_avg_list)
            for key in retrieval_accuracy_avg_list[0]:
                m[key], h[key]  = mean_confidence_interval([float(d[key]) for d in retrieval_accuracy_avg_list])
            retrieval_acc_mean.append(m)
            retrieval_acc_ci.append(h)

            _m, _h = mean_confidence_interval(total_accuracy_avg_list)
            total_acc_mean.append(_m)
            total_acc_ci.append(_h)


    df_atts = {'Method': methods, 
            'KL_Div': kl_mean, 'KL_CI': kl_ci, 
            'MaxSkew': max_skew_mean, 
                'MaxSkew_CI': max_skew_ci,
                }
    if query_is_labeled:
        # df_atts['Worst_Group_Auc_Roc'] = auc_mean
        # df_atts['Worst_Group_Auc_Roc_CI'] = auc_ci
        # df_atts['Gap_CI'] = gap_ci
        df_atts['Clas_Acc'] = classification_accuracy_mean
        df_atts['Clas_Acc_CI'] = classification_accuracy_ci
        df_atts['Retr_Acc'] = retrieval_acc_mean
        df_atts['Retr_Acc_CI'] = retrieval_acc_ci
        df_atts['Tot_Retr_Acc'] = total_acc_mean
        df_atts['Tot_Retr_Acc_CI'] = total_acc_ci
    return df_atts