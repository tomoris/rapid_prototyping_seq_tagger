# -*- coding: utf-8 -*-

neg_tag = 'O'
pad_id = 0


def make_triplet(label_ids, neg_tag, BIES_splitter, index_of_splitted_BIES, label_alphabet):
    triplets_list = []
    for b in range(len(label_ids)):
        triplets = [[-1, -1, -1]]
        for t, tag_id in enumerate(label_ids[b]):
            if label_ids[b][t] == pad_id:
                continue
            tag = label_alphabet[tag_id]
            if tag == neg_tag:
                triplet = [t, t, tag]
                triplets.append(triplet)
            else:
                tag_sp = tag.split(BIES_splitter)
                # try:
                BIES = tag_sp[index_of_splitted_BIES]
                tag_stem = tag_sp[1 - index_of_splitted_BIES]
                # except IndexError:
                #     triplet = [t, t, tag]
                #     triplets.append(triplet)
                #     continue
                if BIES == 'B':
                    triplet = [t, t, tag_stem]
                    triplets.append(triplet)
                elif BIES == 'I':
                    if tag_stem == triplets[-1][2]:
                        triplets[-1][1] = t
                    else:
                        #triplet = [t, t, neg_tag]
                        triplet = [t, t, tag_stem]
                        triplets.append(triplet)
                elif BIES == 'E':
                    if tag_stem == triplets[-1][2]:
                        triplets[-1][1] = t
                    else:
                        #triplet = [t, t, neg_tag]
                        triplet = [t, t, tag_stem]
                        triplets.append(triplet)
                elif BIES == 'S':
                    triplet = [t, t, tag_stem]
                    triplets.append(triplet)
                else:
                    assert(False)
        triplets.pop(0)
        triplets_list.append(triplets)
    return triplets_list


def calc_f_score(tag_seq_RL, batch_label, label_alphabet, beta=1.0, semilabel=True):
    c_lines_triplets = make_triplet(
        batch_label, neg_tag, '-', 0, label_alphabet)
    if semilabel:
        o_lines_triplets = tag_seq_RL
    else:
        o_lines_triplets = make_triplet(
            tag_seq_RL, neg_tag, '-', 0, label_alphabet)
    recall_div = 0
    precision_div = 0
    tp = 0
    recall_tag_div = {}
    precision_tag_div = {}
    tp_tag = {}
    for i in range(len(c_lines_triplets)):
        c_triplets = c_lines_triplets[i]
        o_triplets = o_lines_triplets[i]
        for triplet in c_triplets:
            tag = triplet[2]
            if tag != neg_tag:
                recall_div += 1
                if tag not in recall_tag_div:
                    recall_tag_div[tag] = 0
                    precision_tag_div[tag] = 0
                    tp_tag[tag] = 0
                recall_tag_div[tag] += 1

        for triplet in o_triplets:
            tag = triplet[2]
            if tag != neg_tag:
                precision_div += 1
                if tag not in precision_tag_div:
                    recall_tag_div[tag] = 0
                    precision_tag_div[tag] = 0
                    tp_tag[tag] = 0
                precision_tag_div[tag] += 1
                if triplet in c_triplets:
                    tp += 1
                    tp_tag[tag] += 1

    if precision_div != 0:
        prec = float(tp) / precision_div
    else:
        prec = -1.0
    if recall_div != 0:
        recall = float(tp) / recall_div
    else:
        recall = -1.0
    if prec > 0.0 and recall > 0.0:
        f = (1.0 + beta * beta) * prec * recall / \
            ((beta * beta * prec) + recall)
    else:
        if recall_div == 0 and precision_div == 0:
            f = 1.0
        else:
            f = 0.0

    each_tag_accu = {}
    macro_prec = 0.0
    macro_recall = 0.0
    macro_f = 0.0
    for tag in tp_tag.keys():
        tp_m = tp_tag[tag]
        prec_m_div = precision_tag_div[tag]
        recall_m_div = recall_tag_div[tag]
        if prec_m_div != 0:
            prec_m = float(tp_m) / prec_m_div
        else:
            prec_m = 0.0
        if recall_m_div != 0:
            recall_m = float(tp_m) / recall_m_div
        else:
            recall_m = 0.0
        if prec_m > 0.0 and recall_m > 0.0:
            f_m = (1.0 + beta * beta) * prec_m * recall_m / \
                ((beta * beta * prec_m) + recall_m)
        else:
            if len(tp_tag) == 0:
                f_m = 1.0
            else:
                f_m = 0.0
        each_tag_accu[tag] = [prec_m, recall_m, f_m]
        macro_prec += prec_m
        macro_recall += recall_m
        macro_f += f_m
    if len(tp_tag) != 0:
        macro_prec /= float(len(tp_tag))
        macro_recall /= float(len(tp_tag))
        macro_f /= float(len(tp_tag))
    else:
        macro_prec = 1.0
        macro_recall = 1.0
        macro_f = 1.0

    return prec, recall, f, macro_prec, macro_recall, macro_f, each_tag_accu


def calc_tag_accuracy(tag_seq_RL, batch_label):
    div = 0
    tp = 0
    for b in range(len(tag_seq_RL)):
        for t in range(len(tag_seq_RL[b])):
            if batch_label[b][t] == pad_id:
                assert(tag_seq_RL[b][t] == pad_id)
            else:
                if batch_label[b][t] == tag_seq_RL[b][t]:
                    tp += 1
                div += 1
    accu = float(tp) / float(div)
    return accu


def calc_relaxed_match_f_score(tag_seq_RL, batch_label, label_alphabet, beta=1.0, semilabel=True):
    c_lines_triplets = make_triplet(
        batch_label, neg_tag, '-', 0, label_alphabet)
    if semilabel:
        o_lines_triplets = tag_seq_RL
    else:
        o_lines_triplets = make_triplet(
            tag_seq_RL, neg_tag, '-', 0, label_alphabet)
    recall_div = 0
    precision_div = 0
    tp = 0
    recall_tag_div = {}
    precision_tag_div = {}
    tp_tag = {}
    for i in range(len(c_lines_triplets)):
        c_triplets = c_lines_triplets[i]
        o_triplets = o_lines_triplets[i]
        for triplet in c_triplets:
            tag = triplet[2]
            if tag != neg_tag:
                recall_div += 1
                if tag not in recall_tag_div:
                    recall_tag_div[tag] = 0
                    precision_tag_div[tag] = 0
                    tp_tag[tag] = 0
                recall_tag_div[tag] += 1

        for triplet in o_triplets:
            tag = triplet[2]
            if tag != neg_tag:
                precision_div += 1
                if tag not in precision_tag_div:
                    recall_tag_div[tag] = 0
                    precision_tag_div[tag] = 0
                    tp_tag[tag] = 0
                precision_tag_div[tag] += 1
                start_idx = triplet[0]
                end_idx = triplet[1]
                for c_tri in c_triplets:
                    if tag == c_tri[2]:
                        c_start_idx = c_tri[0]
                        c_end_idx = c_tri[1]
                        if (c_start_idx <= start_idx and start_idx <= c_end_idx) or (c_start_idx <= end_idx and end_idx <= c_end_idx):
                            tp += 1
                            tp_tag[tag] += 1
                            c_triplets.remove(c_tri)
                            break

    if precision_div != 0:
        prec = float(tp) / precision_div
    else:
        prec = -1.0
    if recall_div != 0:
        recall = float(tp) / recall_div
    else:
        recall = -1.0
    if prec > 0.0 and recall > 0.0:
        f = (1.0 + beta * beta) * prec * recall / \
            ((beta * beta * prec) + recall)
    else:
        if recall_div == 0 and precision_div == 0:
            f = 1.0
        else:
            f = 0.0

    each_tag_accu = {}
    macro_prec = 0.0
    macro_recall = 0.0
    macro_f = 0.0
    for tag in tp_tag.keys():
        tp_m = tp_tag[tag]
        prec_m_div = precision_tag_div[tag]
        recall_m_div = recall_tag_div[tag]
        if prec_m_div != 0:
            prec_m = float(tp_m) / prec_m_div
        else:
            prec_m = 0.0
        if recall_m_div != 0:
            recall_m = float(tp_m) / recall_m_div
        else:
            recall_m = 0.0
        if prec_m > 0.0 and recall_m > 0.0:
            f_m = (1.0 + beta * beta) * prec_m * recall_m / \
                ((beta * beta * prec_m) + recall_m)
        else:
            if len(tp_tag) == 0:
                f_m = 1.0
            else:
                f_m = 0.0
        each_tag_accu[tag] = [prec_m, recall_m, f_m]
        macro_prec += prec_m
        macro_recall += recall_m
        macro_f += f_m
    if len(tp_tag) != 0:
        macro_prec /= float(len(tp_tag))
        macro_recall /= float(len(tp_tag))
        macro_f /= float(len(tp_tag))
    else:
        macro_prec = 1.0
        macro_recall = 1.0
        macro_f = 1.0

    return prec, recall, f, macro_prec, macro_recall, macro_f, each_tag_accu


def get_eval_metric_score(tag_seq_RL, batch_label, eval_style, label_alphabet, beta):
    assert(len(tag_seq_RL) == len(batch_label))
    assert(len(tag_seq_RL[0]) == len(batch_label[0]))
    assert(len(tag_seq_RL[-1]) == len(batch_label[-1]))
    if eval_style == 'micro_f':
        p, r, f, _, _, _, _ = calc_f_score(
            tag_seq_RL, batch_label, label_alphabet, beta)
        metric_score = f
    elif eval_style == 'macro_f':
        _, _, _, p, r, f, _ = calc_f_score(
            tag_seq_RL, batch_label, label_alphabet, beta)
        metric_score = f
    elif eval_style == 'relaxed_match_micro_f':
        p, r, f, _, _, _, _ = calc_relaxed_match_f_score(
            tag_seq_RL, batch_label, label_alphabet, beta)
        metric_score = f
    elif eval_style == 'relaxed_match_macro_f':
        _, _, _, p, r, f, _ = calc_relaxed_match_f_score(
            tag_seq_RL, batch_label, label_alphabet, beta)
        metric_score = f
    elif eval_style == 'weighted_macro_f':
        _, _, _, _, _, _, each_tag_accu = calc_f_score(
            tag_seq_RL, batch_label, label_alphabet, beta)
        weights = {'T': 0.12812479168055463, 'F': 0.0963269115392307, 'Ac': 0.09219385374308378, 'Sf': 0.1301246583561096,
                    'Af': 0.13279114725684954, 'D': 0.1394573695086994, 'Q': 0.1397906806212919, 'St': 0.14119058729418038}
        metric_score = 0.0
        for tag, accus in each_tag_accu.items():
            metric_score += weights[tag] * accus[2]
    elif eval_style == 'tag_accuracy':
        metric_score = calc_tag_accuracy(tag_seq_RL, batch_label)
    else:
        assert(False)

    return metric_score

def calc_f_beta(prec, recall, beta):
    if prec > 0.0 and recall > 0.0:
        f = (1.0 + beta * beta) * prec * recall / \
              ((beta * beta * prec) + recall)
    else:
        f = 0.0
    return f


def get_all_scores(all_predict_list, all_correct_list, id2label, id2semilabel, semilabel=True):

    all_correct_list_adjust = []
    for __ in all_correct_list:
        all_correct_list_adjust.append([])
        for _ in __:
            assert(len(_) == 1)
            all_correct_list_adjust[-1].append(_[0])

    if semilabel:
        all_predict_list_adjust = [[ [_[0], _[1], id2semilabel[_[2]]] for _ in __] for __ in all_predict_list]
        assert(len(all_predict_list_adjust) == len(all_correct_list_adjust))
    else:
        all_predict_list_adjust = [[_[2] for _ in __] for __ in all_predict_list]
        assert(len(all_predict_list_adjust) == len(all_correct_list_adjust))
        assert(len(all_predict_list_adjust[0]) == len(all_correct_list_adjust[0]))
        assert(len(all_predict_list_adjust[-1]) == len(all_correct_list_adjust[-1]))
        tag_accu = calc_tag_accuracy(all_predict_list_adjust, all_correct_list_adjust)

    p, r, f, m_p, m_r, m_f, each_tag_accu = calc_f_score(
        all_predict_list_adjust, all_correct_list_adjust, id2label, 1.0, semilabel)

    # weights = {'T': 0.12812479168055463, 'F': 0.0963269115392307, 'Ac': 0.09219385374308378, 'Sf': 0.1301246583561096,
    #            'Af': 0.13279114725684954, 'D': 0.1394573695086994, 'Q': 0.1397906806212919, 'St': 0.14119058729418038}
    # weighted_macro_f = 0.0
    # for tag, accus in each_tag_accu.items():
    #     weighted_macro_f += weights[tag] * accus[2]

    r_p, r_r, r_f, r_m_p, r_m_r, r_m_f, r_each_tag_accu = calc_relaxed_match_f_score(
        all_predict_list_adjust, all_correct_list_adjust, id2label, 1.0, semilabel)

    # r_weighted_macro_f = 0.0
    # for tag, accus in r_each_tag_accu.items():
    #     r_weighted_macro_f += weights[tag] * accus[2]

    f_05 = calc_f_beta(p, r, 0.5)
    f_20 = calc_f_beta(p, r, 2.0)
    m_f_05 = calc_f_beta(m_p, m_r, 0.5)
    m_f_20 = calc_f_beta(m_p, m_r, 2.0)
    r_f_05 = calc_f_beta(r_p, r_r, 0.5)
    r_f_20 = calc_f_beta(r_p, r_r, 2.0)
    r_m_f_05 = calc_f_beta(r_m_p, r_m_r, 0.5)
    r_m_f_20 = calc_f_beta(r_m_p, r_m_r, 2.0)

    all_metrics = {}

    if semilabel is False:
        all_metrics['BIESO_tag_accuracy'] = tag_accu

    all_metrics['micro_precision'] = p
    all_metrics['micro_recall'] = r
    all_metrics['micro_f_1.0'] = f
    all_metrics['micro_f_0.5'] = f_05
    all_metrics['micro_f_2.0'] = f_20

    all_metrics['macro_precision'] = m_p
    all_metrics['macro_recall'] = m_r
    all_metrics['macro_f_1.0'] = m_f
    all_metrics['macro_f_0.5'] = m_f_05
    all_metrics['macro_f_2.0'] = m_f_20

    all_metrics['relaxed_match_micro_precision'] = r_p
    all_metrics['relaxed_match_micro_recall'] = r_r
    all_metrics['relaxed_match_micro_f_1.0'] = r_f
    all_metrics['relaxed_match_micro_f_0.5'] = r_f_05
    all_metrics['relaxed_match_micro_f_2.0'] = r_f_20

    all_metrics['relaxed_match_macro_precision'] = r_m_p
    all_metrics['relaxed_match_macro_recall'] = r_m_r
    all_metrics['relaxed_match_macro_f_1.0'] = r_m_f
    all_metrics['relaxed_match_macro_f_0.5'] = r_m_f_05
    all_metrics['relaxed_match_macro_f_2.0'] = r_m_f_20

    # all_metrics['weighted_macro_f_1.0'] = weighted_macro_f
    # all_metrics['weighted_relaxed_match_macro_f_1.0'] = r_weighted_macro_f

    all_metrics['each_tag_accuracy'] = each_tag_accu
    all_metrics['relaxed_each_tag_accuracy'] = r_each_tag_accu

    return all_metrics
