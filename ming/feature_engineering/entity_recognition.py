"""
医疗实体识别模块
支持心血管、神经内科等专科领域的实体识别
目标：F1值 >= 0.92, 特征提取耗时 <= 50ms/条
"""
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class MedicalEntity:
    """医疗实体数据类"""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class EntityRecognitionResult:
    """实体识别结果"""
    entities: List[MedicalEntity]
    processing_time_ms: float
    coverage_rate: float
    text_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "processing_time_ms": self.processing_time_ms,
            "coverage_rate": self.coverage_rate,
            "text_length": self.text_length
        }


class MedicalEntityRecognizer:
    """
    医疗实体识别器
    
    支持识别的实体类型：
    - DISEASE: 疾病名称
    - SYMPTOM: 症状
    - MEDICINE: 药物
    - BODY_PART: 身体部位
    - EXAMINATION: 检查项目
    - TREATMENT: 治疗方法
    - CLINICAL_DEPT: 临床科室
    - PATHOLOGY: 病理类型
    - DIAGNOSIS: 诊断
    - LAB_VALUE: 实验室指标
    """
    
    ENTITY_TYPES = {
        "DISEASE": "疾病",
        "SYMPTOM": "症状", 
        "MEDICINE": "药物",
        "BODY_PART": "身体部位",
        "EXAMINATION": "检查项目",
        "TREATMENT": "治疗方法",
        "CLINICAL_DEPT": "临床科室",
        "PATHOLOGY": "病理类型",
        "DIAGNOSIS": "诊断",
        "LAB_VALUE": "实验室指标"
    }
    
    SPECIALTY_ENTITIES = {
        "cardiovascular": {
            "DISEASE": [
                "心肌梗死", "心绞痛", "冠心病", "高血压", "心律失常", "心力衰竭",
                "心肌病", "心内膜炎", "心包炎", "先天性心脏病", "风湿性心脏病",
                "主动脉缩窄", "心肌炎", "房颤", "室颤", "心动过速", "心动过缓",
                "亚急性自体瓣膜感染性心内膜炎", "原发性醛固酮增多症"
            ],
            "SYMPTOM": [
                "胸痛", "心悸", "气短", "呼吸困难", "水肿", "发绀", "晕厥",
                "心前区疼痛", "夜间阵发性呼吸困难", "端坐呼吸"
            ],
            "BODY_PART": [
                "心脏", "心房", "心室", "冠状动脉", "主动脉", "二尖瓣", "三尖瓣",
                "左心室", "右心房", "心肌", "心包"
            ],
            "EXAMINATION": [
                "心电图", "超声心动图", "冠脉造影", "心肌酶谱", "BNP", "心脏彩超",
                "Holter监测", "运动平板试验", "心脏CT", "心脏MRI"
            ],
            "MEDICINE": [
                "阿司匹林", "氯吡格雷", "美托洛尔", "硝酸甘油", "华法林", "肝素",
                "ACEI", "ARB", "他汀类", "β受体阻滞剂", "钙通道阻滞剂", "DDAVP"
            ]
        },
        "neurology": {
            "DISEASE": [
                "脑梗死", "脑出血", "癫痫", "帕金森病", "阿尔茨海默病", "多发性硬化",
                "脑膜炎", "脑炎", "脊髓炎", "吉兰-巴雷综合征", "重症肌无力",
                "偏头痛", "紧张性头痛", "三叉神经痛", "面神经麻痹"
            ],
            "SYMPTOM": [
                "头痛", "眩晕", "意识障碍", "抽搐", "肢体无力", "感觉异常",
                "言语障碍", "吞咽困难", "共济失调", "视力障碍", "面瘫"
            ],
            "BODY_PART": [
                "大脑", "小脑", "脑干", "脊髓", "脑膜", "基底节", "丘脑",
                "内囊", "外囊", "脑室", "蛛网膜下腔"
            ],
            "EXAMINATION": [
                "头颅CT", "头颅MRI", "脑电图", "脑脊液检查", "肌电图", "神经传导速度",
                "经颅多普勒", "脑血管造影", "PET-CT"
            ],
            "MEDICINE": [
                "丙戊酸钠", "卡马西平", "左乙拉西坦", "苯妥英钠", "甘露醇",
                "尿激酶", "阿替普酶", "多巴丝肼", "普拉克索"
            ]
        },
        "hematology": {
            "DISEASE": [
                "贫血", "白血病", "淋巴瘤", "血小板减少症", "血友病", "骨髓瘤",
                "特发性血小板减少性紫癜", "ITP", "再生障碍性贫血", "溶血性贫血"
            ],
            "SYMPTOM": [
                "出血", "瘀斑", "淋巴结肿大", "脾大", "肝大", "发热", "乏力"
            ],
            "BODY_PART": [
                "骨髓", "脾脏", "淋巴结", "血液", "血小板", "红细胞", "白细胞"
            ],
            "EXAMINATION": [
                "血常规", "骨髓穿刺", "骨髓活检", "凝血功能", "血涂片",
                "流式细胞术", "染色体核型分析"
            ],
            "MEDICINE": [
                "促红细胞生成素", "G-CSF", "环孢素", "甲氨蝶呤", "阿糖胞苷",
                "长春新碱", "利妥昔单抗"
            ]
        },
        "endocrinology": {
            "DISEASE": [
                "糖尿病", "甲状腺功能亢进", "甲状腺功能减退", "甲状腺炎", "甲状腺结节",
                "肾上腺皮质功能亢进", "肾上腺皮质功能减退", "垂体瘤", "尿崩症",
                "库欣综合征", "艾迪生病"
            ],
            "SYMPTOM": [
                "多饮", "多尿", "多食", "体重下降", "怕热", "怕冷", "乏力",
                "皮肤色素沉着", "向心性肥胖"
            ],
            "BODY_PART": [
                "甲状腺", "肾上腺", "垂体", "胰岛", "甲状旁腺"
            ],
            "EXAMINATION": [
                "血糖", "糖化血红蛋白", "甲状腺功能", "皮质醇", "ACTH",
                "胰岛素释放试验", "OGTT", "甲状腺超声", "垂体MRI"
            ],
            "MEDICINE": [
                "胰岛素", "二甲双胍", "格列美脲", "甲巯咪唑", "丙硫氧嘧啶",
                "左甲状腺素", "氢化可的松", "泼尼松"
            ]
        },
        "gastroenterology": {
            "DISEASE": [
                "胃炎", "消化性溃疡", "肝硬化", "胰腺炎", "炎症性肠病", "克罗恩病",
                "溃疡性结肠炎", "原发性肝癌", "胆囊结石", "胆管炎", "阑尾炎"
            ],
            "SYMPTOM": [
                "腹痛", "恶心", "呕吐", "腹泻", "便秘", "便血", "黄疸", "腹胀"
            ],
            "BODY_PART": [
                "胃", "肠道", "肝脏", "胆囊", "胰腺", "食管", "结肠", "阑尾"
            ],
            "EXAMINATION": [
                "胃镜", "肠镜", "腹部B超", "腹部CT", "肝功能", "甲胎蛋白",
                "淀粉酶", "脂肪酶", "幽门螺杆菌检测"
            ],
            "MEDICINE": [
                "奥美拉唑", "雷贝拉唑", "法莫替丁", "铝碳酸镁", "多潘立酮",
                "莫沙必利", "熊去氧胆酸"
            ]
        },
        "pediatrics": {
            "DISEASE": [
                "麻疹", "风疹", "幼儿急疹", "猩红热", "手足口病", "水痘",
                "百日咳", "流行性腮腺炎", "小儿肺炎", "支气管炎", "营养不良",
                "维生素D缺乏性佝偻病", "风湿热"
            ],
            "SYMPTOM": [
                "发热", "皮疹", "咳嗽", "流涕", "腹泻", "呕吐", "惊厥", "发绀"
            ],
            "BODY_PART": [
                "儿童", "新生儿", "婴幼儿", "青春期"
            ],
            "EXAMINATION": [
                "生长发育评估", "骨龄测定", "新生儿筛查", "微量元素检测"
            ],
            "MEDICINE": [
                "布洛芬", "对乙酰氨基酚", "阿莫西林", "阿奇霉素", "头孢克洛"
            ]
        },
        "obstetrics_gynecology": {
            "DISEASE": [
                "异位妊娠", "输卵管妊娠", "先兆流产", "妊娠期糖尿病", "妊娠期高血压",
                "前置胎盘", "胎盘早剥", "产后出血", "子宫肌瘤", "卵巢囊肿"
            ],
            "SYMPTOM": [
                "停经", "阴道流血", "腹痛", "胎动减少", "下肢水肿"
            ],
            "BODY_PART": [
                "子宫", "卵巢", "输卵管", "宫颈", "胎盘", "胎头", "骨盆"
            ],
            "EXAMINATION": [
                "B超", "胎心监护", "唐氏筛查", "糖耐量试验", "阴道镜", "宫颈涂片"
            ],
            "MEDICINE": [
                "黄体酮", "缩宫素", "米索前列醇", "卡前列素"
            ]
        },
        "psychiatry": {
            "DISEASE": [
                "精神分裂症", "抑郁症", "焦虑症", "双相情感障碍", "强迫症",
                "恐惧症", "癔症", "睡眠障碍"
            ],
            "SYMPTOM": [
                "幻觉", "妄想", "情感淡漠", "意志减退", "焦虑", "抑郁", "失眠",
                "幻视", "幻听", "幻嗅"
            ],
            "BODY_PART": [
                "大脑", "边缘系统", "额叶", "颞叶"
            ],
            "EXAMINATION": [
                "精神检查", "心理量表", "脑电图", "头颅MRI"
            ],
            "MEDICINE": [
                "氯丙嗪", "利培酮", "奥氮平", "喹硫平", "阿立哌唑", "氟西汀",
                "舍曲林", "帕罗西汀", "文拉法辛"
            ]
        },
        "immunology": {
            "DISEASE": [
                "系统性红斑狼疮", "类风湿关节炎", "强直性脊柱炎", "干燥综合征",
                "皮肌炎", "硬皮病", "血管炎"
            ],
            "SYMPTOM": [
                "关节痛", "皮疹", "光敏感", "口干", "眼干", "雷诺现象"
            ],
            "BODY_PART": [
                "关节", "皮肤", "免疫球蛋白", "补体"
            ],
            "EXAMINATION": [
                "ANA", "抗dsDNA抗体", "类风湿因子", "抗CCP抗体", "HLA-B27",
                "免疫球蛋白", "补体C3C4"
            ],
            "MEDICINE": [
                "泼尼松", "甲泼尼龙", "环磷酰胺", "甲氨蝶呤", "来氟米特",
                "羟氯喹", "生物制剂"
            ]
        }
    }
    
    LAB_VALUE_PATTERNS = [
        (r'(\d+\.?\d*)\s*(mmol/L|g/L|mg/dL|U/L|pg/mL|ng/mL|μIU/mL|%)', "LAB_VALUE"),
        (r'(阳性|阴性|\+/-)', "LAB_VALUE"),
        (r'(升高|降低|正常|偏高|偏低)', "LAB_VALUE"),
    ]
    
    CLINICAL_DEPT_PATTERNS = [
        "心血管内科", "神经内科", "血液科", "内分泌科", "消化内科", "儿科",
        "妇产科", "精神科", "免疫科", "肾内科", "呼吸内科", "泌尿外科",
        "普外科", "骨科", "心外科", "神经外科", "肿瘤科", "急诊科", "ICU"
    ]
    
    def __init__(
        self,
        specialty: str = "cardiovascular",
        custom_entity_dict: Optional[Dict[str, List[str]]] = None,
        enable_lab_value_extraction: bool = True
    ):
        """
        初始化医疗实体识别器
        
        Args:
            specialty: 专科领域，支持cardiovascular, neurology, hematology等
            custom_entity_dict: 自定义实体词典
            enable_lab_value_extraction: 是否启用实验室指标提取
        """
        self.specialty = specialty
        self.enable_lab_value_extraction = enable_lab_value_extraction
        
        self._entity_dict = self._build_entity_dict(specialty, custom_entity_dict)
        self._build_reverse_index()
        
    def _build_entity_dict(
        self, 
        specialty: str, 
        custom_entity_dict: Optional[Dict[str, List[str]]]
    ) -> Dict[str, List[str]]:
        """构建实体词典"""
        entity_dict = {}
        
        for spec, entities in self.SPECIALTY_ENTITIES.items():
            for entity_type, entity_list in entities.items():
                if entity_type not in entity_dict:
                    entity_dict[entity_type] = []
                entity_dict[entity_type].extend(entity_list)
        
        if specialty in self.SPECIALTY_ENTITIES:
            for entity_type, entity_list in self.SPECIALTY_ENTITIES[specialty].items():
                if entity_type not in entity_dict:
                    entity_dict[entity_type] = []
                entity_dict[entity_type].extend(entity_list)
        
        if custom_entity_dict:
            for entity_type, entity_list in custom_entity_dict.items():
                if entity_type not in entity_dict:
                    entity_dict[entity_type] = []
                entity_dict[entity_type].extend(entity_list)
        
        for entity_type in entity_dict:
            entity_dict[entity_type] = list(set(entity_dict[entity_type]))
            entity_dict[entity_type].sort(key=len, reverse=True)
            
        return entity_dict
    
    def _build_reverse_index(self) -> None:
        """构建反向索引用于快速匹配"""
        self._entity_set = set()
        self._entity_to_type = {}
        
        for entity_type, entities in self._entity_dict.items():
            for entity in entities:
                self._entity_set.add(entity)
                self._entity_to_type[entity] = entity_type
    
    def recognize(
        self, 
        text: str,
        return_confidence: bool = True
    ) -> EntityRecognitionResult:
        """
        识别文本中的医疗实体
        
        Args:
            text: 输入文本
            return_confidence: 是否返回置信度
            
        Returns:
            EntityRecognitionResult: 识别结果
        """
        start_time = time.time()
        entities = []
        
        for entity_type, entity_list in self._entity_dict.items():
            for entity_text in entity_list:
                start = 0
                while True:
                    pos = text.find(entity_text, start)
                    if pos == -1:
                        break
                    
                    is_overlapping = False
                    for existing in entities:
                        if (pos < existing.end_pos and pos + len(entity_text) > existing.start_pos):
                            if len(entity_text) <= len(existing.text):
                                is_overlapping = True
                                break
                    
                    if not is_overlapping:
                        confidence = 1.0 if return_confidence else None
                        entity = MedicalEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            start_pos=pos,
                            end_pos=pos + len(entity_text),
                            confidence=confidence
                        )
                        entities.append(entity)
                    
                    start = pos + 1
        
        if self.enable_lab_value_extraction:
            lab_entities = self._extract_lab_values(text)
            entities.extend(lab_entities)
        
        dept_entities = self._extract_clinical_depts(text)
        entities.extend(dept_entities)
        
        entities.sort(key=lambda x: x.start_pos)
        
        processing_time = (time.time() - start_time) * 1000
        
        coverage_rate = self._calculate_coverage(text, entities)
        
        return EntityRecognitionResult(
            entities=entities,
            processing_time_ms=processing_time,
            coverage_rate=coverage_rate,
            text_length=len(text)
        )
    
    def _extract_lab_values(self, text: str) -> List[MedicalEntity]:
        """提取实验室指标"""
        entities = []
        for pattern, entity_type in self.LAB_VALUE_PATTERNS:
            for match in re.finditer(pattern, text):
                entity = MedicalEntity(
                    text=match.group(0),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9
                )
                entities.append(entity)
        return entities
    
    def _extract_clinical_depts(self, text: str) -> List[MedicalEntity]:
        """提取临床科室"""
        entities = []
        for dept in self.CLINICAL_DEPT_PATTERNS:
            start = 0
            while True:
                pos = text.find(dept, start)
                if pos == -1:
                    break
                entity = MedicalEntity(
                    text=dept,
                    entity_type="CLINICAL_DEPT",
                    start_pos=pos,
                    end_pos=pos + len(dept),
                    confidence=1.0
                )
                entities.append(entity)
                start = pos + 1
        return entities
    
    def _calculate_coverage(
        self, 
        text: str, 
        entities: List[MedicalEntity]
    ) -> float:
        """计算实体覆盖率"""
        if not text:
            return 0.0
        
        covered_positions = set()
        for entity in entities:
            for pos in range(entity.start_pos, entity.end_pos):
                covered_positions.add(pos)
        
        return len(covered_positions) / len(text)
    
    def get_supported_entity_types(self) -> Dict[str, str]:
        """获取支持的实体类型"""
        return self.ENTITY_TYPES.copy()
    
    def get_entity_count_by_type(
        self, 
        entities: List[MedicalEntity]
    ) -> Dict[str, int]:
        """按类型统计实体数量"""
        count_dict = defaultdict(int)
        for entity in entities:
            count_dict[entity.entity_type] += 1
        return dict(count_dict)
    
    def evaluate(
        self,
        predictions: List[List[MedicalEntity]],
        ground_truths: List[List[MedicalEntity]]
    ) -> Dict[str, float]:
        """
        评估实体识别性能
        
        Args:
            predictions: 预测实体列表
            ground_truths: 真实实体列表
            
        Returns:
            包含precision, recall, f1的字典
        """
        total_pred = 0
        total_gold = 0
        total_correct = 0
        
        for pred_entities, gold_entities in zip(predictions, ground_truths):
            pred_set = set((e.text, e.entity_type, e.start_pos, e.end_pos) 
                          for e in pred_entities)
            gold_set = set((e.text, e.entity_type, e.start_pos, e.end_pos) 
                          for e in gold_entities)
            
            total_pred += len(pred_set)
            total_gold += len(gold_set)
            total_correct += len(pred_set & gold_set)
        
        precision = total_correct / total_pred if total_pred > 0 else 0
        recall = total_correct / total_gold if total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_predictions": total_pred,
            "total_ground_truth": total_gold,
            "total_correct": total_correct
        }
    
    def batch_recognize(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[EntityRecognitionResult]:
        """
        批量识别实体
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度
            
        Returns:
            识别结果列表
        """
        results = []
        texts_iter = texts
        
        if show_progress:
            try:
                from tqdm import tqdm
                texts_iter = tqdm(texts, desc="Recognizing entities")
            except ImportError:
                pass
        
        for text in texts_iter:
            result = self.recognize(text)
            results.append(result)
        
        return results
    
    def get_specialty_entities(self, specialty: str) -> Dict[str, List[str]]:
        """获取特定专科的实体词典"""
        if specialty in self.SPECIALTY_ENTITIES:
            return self.SPECIALTY_ENTITIES[specialty].copy()
        return {}
    
    def add_custom_entities(
        self,
        entity_type: str,
        entities: List[str]
    ) -> None:
        """添加自定义实体"""
        if entity_type not in self._entity_dict:
            self._entity_dict[entity_type] = []
        self._entity_dict[entity_type].extend(entities)
        self._entity_dict[entity_type] = list(set(self._entity_dict[entity_type]))
        self._entity_dict[entity_type].sort(key=len, reverse=True)
        self._build_reverse_index()
