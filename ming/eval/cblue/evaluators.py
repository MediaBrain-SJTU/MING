import json
import time
from sklearn.metrics import classification_report
from transformers import BasicTokenizer
from rouge_chinese import Rouge

from ming.eval.cblue.text2dt_eval_func import text2dt_eval_single_tree

basic_tokenizer = BasicTokenizer(tokenize_chinese_chars=True)


def calc_info_extract_task_scores(list_structured_golden,
                                  list_structured_predict):

    assert len(list_structured_golden) == len(list_structured_predict)

    tp = 0
    fp = 0
    fn = 0
    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):
        assert samp_golden["sample_id"] == samp_predict["sample_id"], "sample ordering is wrong!"
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        assert isinstance(answer_golden, list)
        assert isinstance(answer_predict, list), "sample format is wrong!"

        set_golden = set()
        for inst in answer_golden:
            assert isinstance(inst, dict)
            keys = sorted(list(inst.keys()))
            inst = tuple([json.dumps(inst[w], ensure_ascii=False) for w in keys ])
            # inst = list(inst.items())
            # inst.sort()
            # inst = tuple(inst)

            set_golden.add(inst)

        set_predict = set()
        for inst in answer_predict:
            assert isinstance(inst, dict)
            keys = sorted(list(inst.keys()))
            # inst = tuple([inst[w] for w in keys])
            inst = tuple([json.dumps(inst[w], ensure_ascii=False) for w in keys])

            # inst = list(inst.items())
            # inst.sort()
            # inst = tuple(inst)

            set_predict.add(inst)

        # print("set_predict: ", set_predict)
        # print("set_golden: ", set_golden)

        tp += len(set_golden.intersection(set_predict))
        fp += len(set_predict.difference(set_golden))
        fn += len(set_golden.difference(set_predict))

    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    return precision, recall, f1


def calc_cls_task_scores(list_structured_golden,
                         list_structured_predict,
                         list_labels=None,
                         return_macro=False,
                         ):
    # types = list_labels
    # scores = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in list_labels + ["ALL"]}

    predictions = []
    ground_truths = []

    # Count GT relations and Predicted relations
    assert len(list_structured_golden) == len(list_structured_predict)
    n_sents = len(list_structured_golden)

    # Count TP, FP and FN per type
    for pred_samp, gt_samp in zip(list_structured_predict, list_structured_golden):
        assert pred_samp["sample_id"] == gt_samp["sample_id"], "sample ordering is wrong!"

        pred_label = pred_samp["answer"]
        gt_label = gt_samp["answer"]
        assert gt_label != ""
        if pred_label == "":
            pred_label = list_labels[0]

        predictions.append(pred_label)
        ground_truths.append(gt_label)

    # metric
    t0 = time.time()
    cls_report = classification_report(
        ground_truths, predictions,
        output_dict=True,
        zero_division=0,
    )
    # print(cls_report)

    t1 = time.time()
    # print("calculation metrics: ", t1 - t0)

    if return_macro:
        return cls_report["macro avg"]["precision"], \
               cls_report["macro avg"]["recall"], \
               cls_report["macro avg"]["f1-score"]
    else:
        return cls_report["weighted avg"]["precision"], \
               cls_report["weighted avg"]["recall"], \
               cls_report["weighted avg"]["f1-score"]


def calc_nlg_task_scores(list_structured_golden, list_structured_predict):


    assert len(list_structured_golden) == len(list_structured_predict)

    scores = []
    predictions = []
    references = []
    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):
        # print("samp_golden: ", samp_golden)
        # print("samp_predict: ", samp_predict)

        assert samp_golden["sample_id"] == samp_predict["sample_id"], "sample ordering is wrong!"
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        assert isinstance(answer_golden, str)
        assert isinstance(answer_predict, str), "sample format is wrong!"

        # basic tokenizer: 拆分中文字，保留英文单词
        answer_predict = basic_tokenizer.tokenize(answer_predict)
        answer_golden = basic_tokenizer.tokenize(answer_golden)
        answer_predict = " ".join(answer_predict).strip()
        answer_golden = " ".join(answer_golden).strip()
        if answer_golden.strip() == "":
            answer_golden = "无 。"
        if answer_predict.strip() == "":
            answer_predict = "无 。"
        # print("answer_predict: ", answer_predict)
        # print("answer_golden: ", answer_golden)

        predictions.append(answer_predict)
        references.append(answer_golden)

    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)

    rouge1 = scores["rouge-1"]["f"]
    rouge2 = scores["rouge-2"]["f"]
    rougeL = scores["rouge-l"]["f"]

    return rouge1, rouge2, rougeL


def calc_nlg_task_scores_by_sessions(list_structured_golden, list_structured_predict):


    assert len(list_structured_golden) == len(list_structured_predict)

    scores = []
    predictions = []
    references = []
    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):
        # print("samp_golden: ", samp_golden)
        # print("samp_predict: ", samp_predict)

        assert samp_golden["sample_id"] == samp_predict["sample_id"], "sample ordering is wrong!"
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        # if set(answer_golden.keys()) != set(answer_predict.keys())

        for key in answer_golden.keys():
            pred = answer_predict.get(key, "").strip()
            gt = answer_golden[key].strip()

            # basic tokenizer: 拆分中文字，保留英文单词
            pred = basic_tokenizer.tokenize(pred)
            gt = basic_tokenizer.tokenize(gt)
            pred = " ".join(pred).strip()
            gt = " ".join(gt).strip()
            if gt.strip() == "":
                gt = "无 。"
            if pred.strip() == "":
                pred = "无 。"

            # if gt != pred:
            #     print(gt)
            #     print(pred)

            predictions.append(
                pred
            )
            references.append(
                gt
            )

    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    rouge1 = scores["rouge-1"]["f"]
    rouge2 = scores["rouge-2"]["f"]
    rougeL = scores["rouge-l"]["f"]

    return rouge1, rouge2, rougeL


def calc_text2dt_task_scores(list_structured_golden,
                         list_structured_predict,):

    assert len(list_structured_golden) == len(list_structured_predict)

    gold_tree_num, correct_tree_num = 0.000001, 0.000001
    gold_triplet_num, predict_triplet_num, correct_triplet_num = 0.000001, 0.000001, 0.000001
    gold_path_num, predict_path_num, correct_path_num = 0.000001, 0.000001, 0.000001
    gold_node_num, predict_node_num, correct_node_num = 0.000001, 0.000001, 0.000001

    edit_dis = 0
    max_edit_dis = 0

    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):
        assert samp_golden["sample_id"] == samp_predict["sample_id"], "sample ordering is wrong!"
        tree_golden = samp_golden["answer"]
        tree_predict = samp_predict["answer"]

        assert isinstance(tree_golden, list)
        assert isinstance(tree_predict, list), "sample format is wrong!"

        tmp = text2dt_eval_single_tree(tree_predict, tree_golden)
        gold_tree_num += tmp[0]
        correct_tree_num += tmp[1]
        correct_triplet_num += tmp[2]
        predict_triplet_num += tmp[3]
        gold_triplet_num += tmp[4]
        correct_path_num += tmp[5]
        predict_path_num += tmp[6]
        gold_path_num += tmp[7]
        edit_dis += tmp[8]

        # 计算最大编辑数
        max_edit_dis += (tmp[3] + tmp[10] * 2) + (tmp[4] + tmp[11] * 2)

        correct_node_num += tmp[9]
        predict_node_num += tmp[10]
        gold_node_num += tmp[11]

    tree_acc = correct_tree_num / gold_tree_num
    triplet_f1 = 2 * (correct_triplet_num / predict_triplet_num) * (correct_triplet_num / gold_triplet_num) / (
                correct_triplet_num / predict_triplet_num + correct_triplet_num / gold_triplet_num)
    path_f1 = 2 * (correct_path_num / predict_path_num) * (correct_path_num / gold_path_num) / (
                correct_path_num / predict_path_num + correct_path_num / gold_path_num)
    tree_lenv_radio = 1 - edit_dis / max_edit_dis
    node_f1 = 2 * (correct_node_num / predict_node_num) * (correct_node_num / gold_node_num) / (
                correct_node_num / predict_node_num + correct_node_num / gold_node_num)

    return tree_lenv_radio, node_f1, path_f1