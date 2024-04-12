# coding=utf-8
# Created by Michael Zhu
# ECNU, 2023

import json
import sys

from tqdm import tqdm


def process_generated_results(questions):

    structured_output = {
        "CMeEE-V2": [],
        "CMeIE": [],
        "CHIP-CDN": [],
        "CHIP-CDEE": [],
        "CHIP-STS": [],
        "CHIP-CTC": [],
        "CHIP-MDCFNPC": [],
        "KUAKE-IR": [],
        "KUAKE-QIC": [],
        "KUAKE-QQR": [],
        "KUAKE-QTR": [],
        "MedDG": [],
        "IMCS-V2-MRG": [],
        "IMCS-V2-NER": [],
        "IMCS-V2-DAC": [],
        "IMCS-V2-SR": [],
        "Text2DT": [],
        "CMedCausal": [],
    }

    # with open(pred_file, "r", encoding="utf-8") as f:
    for line in tqdm(questions, ncols=60):
        # line = json.loads(line)
        # print("line: ", line)

        sample_id_ = line["additional_info"].get("sample_id", "xxxx")
        input = line["prompt"]
        gen_output = line["text"]
        # gen_output = gen_output.replace(":", "：", 100).replace(",", "，", 100).replace(";", "；", 100)
        # gen_output = line["generated_output"]
        task_dataset = line["additional_info"]["task_dataset"]
        task_type = line["additional_info"]["task_type"]

        # 选项：
        answer_choices = line["additional_info"]["answer_choices"]

        if task_dataset == "CMeEE-V2":
            # 答案格式：
            #   第一行：引导词
            #   实体每类占一行，每行格式为 "[类型名称]实体：实体名称1，实体名称2，实体名称3\n"
            #                多个实体，用 ， 符号分割

            list_entities = []
            assert isinstance(answer_choices, list)
            for choice in answer_choices:
                for piece in gen_output.split("\n"):
                    if piece.startswith(f"{choice}实体"):
                        mentions = piece.replace(f"{choice}实体：", "").split("，")
                        mentions = [w.strip() for w in mentions if len(w.strip()) > 0]
                        for ment in mentions:
                            list_entities.append(
                                {
                                    "entity": ment,
                                    "type": choice,
                                    # "sample_id": sample_id_,
                                }
                            )

            # print("line: ", line)
            # print("gen_output: ", gen_output)
            # print("list_entities: ", list_entities)
            structured_output["CMeEE-V2"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_entities,
                }
            )

        elif task_dataset == "CMeIE":
            # 答案格式：
            #   每个关系类型占一行，格式为
            #         "具有{lab}关系的实体对如下：头实体：str，尾实体：str；头实体：str，尾实体：str；"

            list_spos = []
            assert isinstance(answer_choices, list)
            list_answer_strs = gen_output.split("\n")

            for line in list_answer_strs:
                # print("line: ", line)
                # 首先是解析出label:
                predicate = line.split("关系的头尾实体对")[0][2: ].strip()
                # print("predicate: ", predicate)
                line = line.replace(f"具有{predicate}关系的头尾实体对如下：", "")

                for spo_str in line.split("。"):
                    # print("spo_str: ", spo_str)

                    if len(spo_str.split("，尾实体为")) < 2:
                        continue

                    head_mention_str, tail_mention_str = spo_str.split("，尾实体为")[:2]
                    head_mention_str = head_mention_str.replace("头实体为", "").strip()
                    tail_mention_str = tail_mention_str.replace("尾实体为", "").strip()
                    # print("head_mention_str: ", head_mention_str)
                    # print("tail_mention_str: ", tail_mention_str)

                    list_spos.append(
                        {
                                "predicate": predicate,
                                "subject": head_mention_str,
                                "object": tail_mention_str,
                        }
                    )

            # print("line: ", line)
            # print("gen_output: ", gen_output)
            # print("list_spos: ", list_spos)

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_spos,
                }
            )

        elif task_dataset == "CMedCausal":

            list_spos = []
            list_answer_strs = gen_output.split("\n")

            rel = None
            for ans_str in list_answer_strs:
                ans_str = ans_str.strip()
                if "因果关系三元组如下" in ans_str:
                    rel = "因果关系"
                    continue
                if "条件关系三元组" in ans_str:
                    rel = "条件关系"
                    continue
                if "上下位关系" in ans_str:
                    rel = "上下位关系"
                    continue

                if "给定的文本没有指定关系的三元组" in ans_str:
                    break

                if not ans_str:
                    continue

                # print("ans_str: ", ans_str)
                head_ent = ans_str.split("头实体：")[1].split("；尾实体：")[0]
                if rel != "条件关系":
                    tail_ent = ans_str.split("；尾实体：")[1]
                    triple = {
                        "predicate": rel,
                        "subject": head_ent,
                        "object": tail_ent,
                    }

                else:
                    tail_tuple_str = ans_str.split("；尾三元组：")[1]
                    # 头实体：xxx；尾实体：xxx；关系：因果关系
                    head_ent_1 = tail_tuple_str.split("头实体：")[1].split("；尾实体：")[0]
                    tail_ent_1 = tail_tuple_str.split("；尾实体：")[1].split("；关系：")[0]
                    rel_1 = tail_tuple_str.split("；关系：")[1].split("；关系：")[0]
                    assert rel_1 == "因果关系"
                    tail_triple = {
                                "predicate": rel_1,
                                "subject": head_ent_1,
                                "object": tail_ent_1,
                        }

                    triple = {
                        "predicate": rel,
                        "subject": head_ent,
                        "object": tail_triple,
                    }

                list_spos.append(triple)

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_spos,
                }
            )


        elif task_dataset == "Text2DT":

            list_nodes = []
            list_answer_strs = gen_output.split("\n")

            for ans_str in list_answer_strs:
                if "根据给定的指南文本抽取的诊疗决策树" in ans_str:
                    continue

                if not ans_str:
                    continue

                role = ans_str.split("：role=")[1].split("；logical_rel=")[0]
                print(ans_str)

                try:
                    logical_rel = ans_str.split("；logical_rel=")[1].split("；triples=")[0]
                except Exception as e:
                    logical_rel = "null"

                try:
                    triples = ans_str.split("；triples=")[1]
                    triples = eval(triples)
                    if not isinstance(triples, list):
                        triples = []

                    triples_new = []
                    for tri in triples:
                        if len(tri) != 3:
                            continue
                        triples_new.append(tri)

                except Exception as e:
                    print(e)
                    triples_new = []

                node = {
                    "role": role,
                    "logical_rel": logical_rel,
                    "triples": triples_new,
                }
                list_nodes.append(node)

            if not list_nodes:
                list_nodes = [
                    {
                        "role": "D",
                        "logical_rel": "null",
                        "triples": []
                    }
                ]

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_nodes,
                }
            )


        elif task_dataset == "CHIP-CDN":
            # 答案格式：
            #   多个选中的标准化实体，用 ， 符号分割

            answer_str = gen_output.split("\n")[-1]
            answers = answer_str.split("，")
            answers = [w.strip() for w in answers if len(w.strip()) > 0]
            #

            answers = [w for w in answers if w in answer_choices]
            answers = list(set(answers))
            answers = [
                {
                    "entity": w,
                    "type": "normalization",
                    # "sample_id": sample_id_,
                }
                for w in answers
            ]
            # print("line: ", line)
            # print("gen_output: ", gen_output)
            # print("answers: ", answers)

            structured_output["CHIP-CDN"].append(
                {
                    "sample_id": sample_id_,
                    "answer": answers,
                }
            )

        elif task_dataset == "CHIP-CDEE":
            # 答案格式：
            #   第一行：引导词
            #   每个事件占一行，事件字段用 ； 分隔， 然后每个字段是 字段名：字段值的格式"
            #                                  字段值有多个，则用 ，符号分隔
            keys = ["主体词", "发生状态", "描述词", "解剖部位"]

            list_answer_strs = gen_output.split("\n")[1: ]
            list_events = []
            for ans_str in list_answer_strs:
                event_info = {}
                ans_attrs = ans_str.split("；")
                for a_attr in ans_attrs:
                    # print("a_attr: ", a_attr)
                    for key in keys:
                        if a_attr.startswith(f"{key}："):
                            a_attr = a_attr.replace(f"{key}：", "").strip()
                            if key in ["描述词", "解剖部位"]:
                                a_attr_split = a_attr.split("，")
                                a_attr_split = [w.strip() for w in a_attr_split if len(w.strip()) > 0]
                                event_info[key] = a_attr_split
                            else:
                                event_info[key] = a_attr

                for key in keys:
                    if key not in event_info:
                        if key in ["描述词", "解剖部位"]:
                            event_info[key] = []
                        else:
                            event_info[key] = ""

                # event_info["sample_id"] = sample_id_
                list_events.append(event_info)

            # print("line: ", line)
            # print("gen_output: ", gen_output)
            # print("list_events: ", list_events)

            structured_output["CHIP-CDEE"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_events,
                }
            )

        elif task_dataset == "CHIP-STS":
            # 答案格式：直接回答"是"，"不是"，"相同"， ”不同“
            answer_str = gen_output.strip()
            # if answer_str not in answer_choices:
            #     answer_str = "不是"

            answer_choices = ["是的", "不是", ]
            if answer_str == "相同":
                answer_str = "是的"
            elif answer_str == "不同":
                answer_str = "不是"

            # print(answer_str)
            # if answer_str not in answer_choices:
            #     answer_str = "不是"

            structured_output["CHIP-STS"].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_str,
                }
            )

        elif task_dataset == "CHIP-CTC":
            # 答案格式：直接回答分类标签
            answer_str = gen_output.strip()
            # if not answer_str in answer_choices:
            #     answer_str = "非上述类型"

            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_str,
                }
            )

        elif task_dataset == "KUAKE-IR":
            # 答案格式：直接回答 "相关", "不相关"
            answer_str = gen_output.strip()
            # if answer_str not in answer_choices:
            #     answer_str = "不相关"

            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_str,
                }
            )

        elif task_dataset == "KUAKE-QIC":
            # 答案格式：直接回答分类标签
            answer_str = gen_output.strip()
            # if not answer_str in answer_choices:
            #     answer_str = "非上述类型"

            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_str,
                }
            )

        elif task_dataset == "KUAKE-QQR":
            # 答案格式：直接回答分类标签
            answer_str = gen_output.strip()
            # if not answer_str in answer_choices:
            #     answer_str = "后者是前者的语义父集或语义毫无关联"


            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_str,
                }
            )

        elif task_dataset == "KUAKE-QTR":
            # 答案格式：直接回答分类标签
            answer_str = gen_output.strip()
            # if not answer_str in answer_choices:
            #     answer_str = "完全不匹配"


            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_str,
                }
            )

        elif task_dataset == "CHIP-MDCFNPC":
            # 答案格式：
            #   第一行：引导词
            #    每一行就是 "[症状词]：[阴阳性判断结果]"
            list_answer_strs = gen_output.split("\n")[1:]

            list_finding_attrs = []
            for ans_str in list_answer_strs:
                if not len(ans_str.split("：")) == 2:
                    continue

                finding, conclusion = ans_str.split("：")
                if conclusion not in answer_choices:
                    conclusion = "不标注"

                list_finding_attrs.append(
                    {
                        "entity": finding.strip(),
                        "attr": conclusion
                    }
                )

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_finding_attrs,
                }
            )

        elif task_dataset == "IMCS-V2-NER":
            # 答案格式：
            #   第一行：引导词
            #   实体每类占一行，每行格式为 "[类型名称]实体：实体名称1，实体名称2，实体名称3\n"
            #                多个实体，用 ， 符号分割

            list_entities = []
            assert isinstance(answer_choices, list)
            for choice in answer_choices:
                for piece in gen_output.split("\n"):
                    if piece.startswith(f"{choice}实体"):
                        mentions = piece.replace(f"{choice}实体：", "").split("，")
                        mentions = [w.strip() for w in mentions if len(w.strip()) > 0]
                        for ment in mentions:
                            list_entities.append(
                                {
                                    "entity": ment,
                                    "type": choice,
                                }
                            )

            structured_output["IMCS-V2-NER"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_entities,
                }
            )

        elif task_dataset == "IMCS-V2-DAC":
            # 答案格式：直接回答分类标签
            answer_str = gen_output.strip()
            # if not answer_str in answer_choices:
            #     answer_str = "非上述类型"

            # print("line: ", line)
            # print("gen_output: ", gen_output)
            # print("answer_str: ", answer_str)

            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_str,
                }
            )

        elif task_dataset == "IMCS-V2-SR":
            # 答案格式：
            #   第一行：引导词
            #    每一行就是 "[症状词]：[阴阳性判断结果]"
            list_answer_strs = gen_output.split("\n")[1:]

            list_finding_attrs = []
            for ans_str in list_answer_strs:
                if not len(ans_str.split("：")) == 2:
                    continue

                finding, conclusion = ans_str.split("：")
                if conclusion not in answer_choices:
                    conclusion = "无法根据上下文确定病人是否患有该症状"

                list_finding_attrs.append(
                    {
                        "entity": finding.strip(),
                        "attr": conclusion
                    }
                )

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_finding_attrs,
                }
            )

        elif task_dataset == "IMCS-V2-MRG":
            # 答案格式：
            #   1. 第一行是引导词；
            #   第二行开始是 [section_name]：str的格式

            keys = [
                "主诉：",
                "现病史：",
                "辅助检查：",
                "既往史：",
                "诊断：",
                "建议："
            ]
            answer_dict = {}
            for key in keys:
                for line in gen_output.strip().split("\n")[1: ]:
                    # print("line: ", line)
                    if not line.startswith(key):
                        continue
                    answer_str = line.strip().split(key)[-1].strip()
                    answer_dict[key[: -1]] = answer_str

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_dict,
                }
            )

        elif task_dataset == "MedDG":
            # 答案格式：str
            answer_str = gen_output.strip()

            # print("line: ", line)
            # print("gen_output: ", gen_output)
            # print("answer_str: ", answer_str)

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": answer_str,
                }
            )




        else:
            # print("task_dataset: ", task_dataset)
            # print("task_type: ", task_type)

            raise ValueError


    return structured_output


if __name__ == "__main__":
    from_dir = sys.argv[1]
    to_dir = sys.argv[2]

    structured_outputs = process_generated_results(
        from_dir
    )
    for key in structured_outputs.keys():
        print(key, len(structured_outputs[key]))


    json.dump(
        structured_outputs,
        open(to_dir, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2
    )