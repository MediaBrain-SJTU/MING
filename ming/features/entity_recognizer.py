"""
医疗实体识别模块

提供基于规则和轻量级模型的医疗实体识别功能，
支持疾病、症状、药物、检查等多种实体类型。
"""

import re
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
import json


class EntityType(Enum):
    """医疗实体类型枚举"""
    DISEASE = auto()      # 疾病
    SYMPTOM = auto()      # 症状
    DRUG = auto()         # 药物
    EXAM = auto()         # 检查/检验
    BODY = auto()         # 身体部位
    TREATMENT = auto()    # 治疗/手术
    DEPARTMENT = auto()   # 科室
    SPECIALTY = auto()    # 专科领域


@dataclass
class MedicalEntity:
    """医疗实体数据结构"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    normalized_form: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "text": self.text,
            "entity_type": self.entity_type.name,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "normalized_form": self.normalized_form or self.text
        }


class MedicalEntityRecognizer:
    """
    医疗实体识别器
    
    基于规则字典和模式匹配的医疗实体识别，
    针对中文医疗文本优化，支持专科领域实体识别。
    
    Attributes:
        entity_dicts: 各类实体的词典
        pattern_rules: 正则表达式规则
        specialty_keywords: 专科关键词映射
    
    Example:
        >>> recognizer = MedicalEntityRecognizer()
        >>> text = "患者患有高血压，需要服用降压药"
        >>> entities = recognizer.recognize(text)
    """
    
    def __init__(self, custom_dict_path: Optional[str] = None):
        """
        初始化实体识别器
        
        Args:
            custom_dict_path: 自定义词典路径，可选
        """
        self.entity_dicts: Dict[EntityType, Set[str]] = {}
        self.pattern_rules: Dict[EntityType, List[str]] = {}
        self.specialty_keywords: Dict[str, List[str]] = {}
        
        self._init_default_dicts()
        self._init_pattern_rules()
        self._init_specialty_keywords()
        
        if custom_dict_path:
            self._load_custom_dict(custom_dict_path)
    
    def _init_default_dicts(self) -> None:
        """初始化默认医疗词典"""
        # 疾病词典
        self.entity_dicts[EntityType.DISEASE] = {
            "高血压", "糖尿病", "冠心病", "心肌梗死", "心绞痛",
            "心律失常", "心力衰竭", "心肌炎", "心包炎", "先天性心脏病",
            "脑卒中", "脑梗死", "脑出血", "癫痫", "帕金森病",
            "阿尔茨海默病", "多发性硬化", "重症肌无力", "格林巴利综合征",
            "肺炎", "肺结核", "慢性阻塞性肺病", "哮喘", "肺癌",
            "胃炎", "胃溃疡", "胃癌", "肝炎", "肝硬化", "肝癌",
            "肾炎", "肾衰竭", "肾结石", "尿路感染",
            "贫血", "白血病", "淋巴瘤", "血小板减少症",
            "甲状腺功能亢进", "甲状腺功能减退", "甲状腺结节",
            "类风湿关节炎", "系统性红斑狼疮", "强直性脊柱炎",
            "抑郁症", "焦虑症", "精神分裂症", "双相情感障碍",
            "白内障", "青光眼", "黄斑变性", "视网膜脱离",
            "中耳炎", "鼻炎", "鼻窦炎", "扁桃体炎",
            "骨折", "骨质疏松", "关节炎", "椎间盘突出",
            "肿瘤", "癌症", "恶性肿瘤", "良性肿瘤",
        }
        
        # 症状词典
        self.entity_dicts[EntityType.SYMPTOM] = {
            "头痛", "头晕", "恶心", "呕吐", "腹痛", "腹泻",
            "发热", "咳嗽", "咳痰", "胸痛", "胸闷", "心悸",
            "呼吸困难", "气促", "乏力", "消瘦", "水肿",
            "失眠", "嗜睡", "昏迷", "抽搐", "瘫痪",
            "出血", "淤血", "黄疸", "皮疹", "瘙痒",
            "视力模糊", "听力下降", "耳鸣", "鼻塞", "流涕",
            "关节痛", "肌肉痛", "腰痛", "背痛", "颈痛",
            "食欲不振", "消化不良", "便秘", "便血", "黑便",
            "尿频", "尿急", "尿痛", "血尿", "少尿", "多尿",
            "焦虑", "抑郁", "幻觉", "妄想", "记忆力下降",
        }
        
        # 药物词典
        self.entity_dicts[EntityType.DRUG] = {
            "阿司匹林", "氯吡格雷", "华法林", "利伐沙班",
            "阿托伐他汀", "瑞舒伐他汀", "辛伐他汀",
            "硝苯地平", "氨氯地平", "缬沙坦", "厄贝沙坦",
            "美托洛尔", "比索洛尔", "卡维地洛",
            "呋塞米", "氢氯噻嗪", "螺内酯",
            "二甲双胍", "格列美脲", "胰岛素",
            "奥美拉唑", "泮托拉唑", "雷贝拉唑",
            "头孢", "青霉素", "阿莫西林", "左氧氟沙星",
            "布洛芬", "对乙酰氨基酚", "塞来昔布",
            "地西泮", "劳拉西泮", "阿普唑仑",
            "多巴丝肼", "卡比多巴", "左旋多巴",
            "硝酸甘油", "单硝酸异山梨酯", "地高辛",
        }
        
        # 检查/检验词典
        self.entity_dicts[EntityType.EXAM] = {
            "血常规", "尿常规", "便常规", "生化全套",
            "肝功能", "肾功能", "血脂", "血糖", "糖化血红蛋白",
            "心电图", "动态心电图", "运动平板试验",
            "超声心动图", "心脏彩超", "胸部X线", "胸部CT",
            "头颅CT", "头颅MRI", "脑血管造影", "颈动脉超声",
            "胃镜", "肠镜", "支气管镜", "膀胱镜",
            "肺功能", "血气分析", "肿瘤标志物",
            "凝血功能", "D-二聚体", "心肌酶谱", "肌钙蛋白",
            "脑电图", "肌电图", "神经传导速度",
            "骨密度", "骨扫描", "PET-CT", "病理活检",
        }
        
        # 身体部位词典
        self.entity_dicts[EntityType.BODY] = {
            "头部", "颈部", "胸部", "腹部", "背部", "腰部",
            "上肢", "下肢", "手臂", "腿部", "手部", "足部",
            "心脏", "肺", "肝", "肾", "脾", "胃", "肠",
            "大脑", "小脑", "脑干", "脊髓", "神经",
            "眼睛", "耳朵", "鼻子", "口腔", "咽喉",
            "骨骼", "肌肉", "关节", "皮肤",
            "血管", "动脉", "静脉", "毛细血管",
            "甲状腺", "肾上腺", "胰腺", "胆囊",
        }
        
        # 治疗/手术词典
        self.entity_dicts[EntityType.TREATMENT] = {
            "手术", "切除术", "移植术", "搭桥术", "支架植入",
            "化疗", "放疗", "靶向治疗", "免疫治疗",
            "透析", "输血", "输液", "注射",
            "物理治疗", "康复治疗", "心理治疗",
            "介入治疗", "微创手术", "开腹手术",
            "冠状动脉造影", "PCI", "CABG",
            "起搏器植入", "除颤器植入", "射频消融",
        }
        
        # 科室词典
        self.entity_dicts[EntityType.DEPARTMENT] = {
            "心内科", "心外科", "神经内科", "神经外科",
            "呼吸内科", "消化内科", "肾内科", "内分泌科",
            "血液科", "风湿免疫科", "感染科", "肿瘤科",
            "普外科", "骨科", "泌尿外科", "胸外科",
            "妇产科", "儿科", "眼科", "耳鼻喉科",
            "口腔科", "皮肤科", "精神科", "心理科",
            "急诊科", "ICU", "康复科", "中医科",
            "放射科", "超声科", "检验科", "病理科",
        }
    
    def _init_pattern_rules(self) -> None:
        """初始化正则表达式规则"""
        # 疾病模式
        self.pattern_rules[EntityType.DISEASE] = [
            r"[急慢]性\w+[炎病]",
            r"\w+综合征",
            r"\w+型\w+病",
            r"第[一二三四五]期\w+",
        ]
        
        # 症状模式
        self.pattern_rules[EntityType.SYMPTOM] = [
            r"\w+痛",
            r"\w+肿",
            r"\w+热",
            r"\w+困难",
        ]
        
        # 药物模式
        self.pattern_rules[EntityType.DRUG] = [
            r"\w+片",
            r"\w+胶囊",
            r"\w+注射液",
            r"\w+颗粒",
        ]
        
        # 检查模式
        self.pattern_rules[EntityType.EXAM] = [
            r"\w+检查",
            r"\w+检验",
            r"\w+造影",
            r"\w+扫描",
        ]
    
    def _init_specialty_keywords(self) -> None:
        """初始化专科关键词映射"""
        self.specialty_keywords = {
            "心血管": [
                "心脏", "血管", "动脉", "静脉", "血压", "心率",
                "冠心病", "心肌梗死", "心绞痛", "心律失常", "心力衰竭",
                "心电图", "超声心动图", "支架", "搭桥", "起搏器"
            ],
            "神经内科": [
                "脑", "神经", "脊髓", "癫痫", "帕金森", "卒中",
                "脑梗", "脑出血", "头痛", "头晕", "昏迷", "瘫痪",
                "脑电图", "肌电图", "腰穿"
            ],
            "呼吸": [
                "肺", "支气管", "气管", "呼吸", "咳嗽", "咳痰",
                "哮喘", "肺炎", "肺结核", "慢阻肺", "肺癌",
                "肺功能", "支气管镜", "氧疗"
            ],
            "消化": [
                "胃", "肠", "肝", "胆", "胰", "脾",
                "胃炎", "溃疡", "肝炎", "肝硬化", "胰腺炎",
                "胃镜", "肠镜", "腹部超声"
            ],
            "内分泌": [
                "甲状腺", "肾上腺", "垂体", "胰腺",
                "糖尿病", "甲亢", "甲减", "肥胖", "骨质疏松",
                "血糖", "胰岛素", "激素"
            ],
        }
    
    def _load_custom_dict(self, dict_path: str) -> None:
        """
        加载自定义词典
        
        Args:
            dict_path: 词典文件路径
        """
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                custom_dict = json.load(f)
                for entity_type_name, terms in custom_dict.items():
                    try:
                        entity_type = EntityType[entity_type_name.upper()]
                        if entity_type in self.entity_dicts:
                            self.entity_dicts[entity_type].update(set(terms))
                        else:
                            self.entity_dicts[entity_type] = set(terms)
                    except KeyError:
                        continue
        except Exception as e:
            print(f"加载自定义词典失败: {e}")
    
    def recognize(self, text: str) -> List[MedicalEntity]:
        """
        识别文本中的医疗实体
        
        Args:
            text: 输入文本
            
        Returns:
            识别出的医疗实体列表
            
        Example:
            >>> recognizer = MedicalEntityRecognizer()
            >>> entities = recognizer.recognize("患者有高血压病史")
            >>> print(entities[0].text)
            '高血压'
        """
        start_time = time.time()
        entities: List[MedicalEntity] = []
        
        # 基于词典的匹配
        for entity_type, term_set in self.entity_dicts.items():
            for term in term_set:
                for match in re.finditer(re.escape(term), text):
                    entity = MedicalEntity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=1.0
                    )
                    entities.append(entity)
        
        # 基于规则的匹配
        for entity_type, patterns in self.pattern_rules.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    # 避免重复
                    is_duplicate = False
                    for existing in entities:
                        if (match.start() >= existing.start_pos and 
                            match.end() <= existing.end_pos):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        entity = MedicalEntity(
                            text=match.group(),
                            entity_type=entity_type,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=0.8
                        )
                        entities.append(entity)
        
        # 按位置排序并去重
        entities = sorted(entities, key=lambda x: (x.start_pos, -x.end_pos))
        filtered_entities: List[MedicalEntity] = []
        for entity in entities:
            overlap = False
            for existing in filtered_entities:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    overlap = True
                    break
            if not overlap:
                filtered_entities.append(entity)
        
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        return filtered_entities
    
    def recognize_by_specialty(self, text: str, specialty: str) -> List[MedicalEntity]:
        """
        针对特定专科领域进行实体识别
        
        Args:
            text: 输入文本
            specialty: 专科名称（如"心血管", "神经内科"）
            
        Returns:
            该专科相关的实体列表
        """
        all_entities = self.recognize(text)
        
        if specialty not in self.specialty_keywords:
            return all_entities
        
        specialty_terms = set(self.specialty_keywords[specialty])
        
        # 过滤与专科相关的实体
        specialty_entities = []
        for entity in all_entities:
            if entity.text in specialty_terms:
                entity.confidence = min(1.0, entity.confidence + 0.1)
                specialty_entities.append(entity)
        
        return specialty_entities
    
    def get_entity_coverage(self, texts: List[str]) -> Dict[str, float]:
        """
        计算实体类型覆盖率
        
        Args:
            texts: 文本列表
            
        Returns:
            各实体类型的覆盖率统计
        """
        entity_type_counts: Dict[EntityType, int] = {et: 0 for et in EntityType}
        total_texts = len(texts)
        
        for text in texts:
            entities = self.recognize(text)
            found_types = set(e.entity_type for e in entities)
            for et in found_types:
                entity_type_counts[et] += 1
        
        coverage = {
            et.name: count / total_texts 
            for et, count in entity_type_counts.items()
        }
        
        return coverage
    
    def batch_recognize(self, texts: List[str]) -> List[List[MedicalEntity]]:
        """
        批量识别文本中的实体
        
        Args:
            texts: 文本列表
            
        Returns:
            每个文本的实体列表
        """
        return [self.recognize(text) for text in texts]
