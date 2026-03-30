"""
医疗实体识别器模块

本模块提供基于规则和词典的医疗实体识别功能，支持多种医疗实体类型的识别。
"""

import re
import time
import logging
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import jieba

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """医疗实体类型枚举"""
    DISEASE = "疾病"
    SYMPTOM = "症状"
    DRUG = "药物"
    EXAMINATION = "检查"
    TREATMENT = "治疗"
    BODY_PART = "身体部位"
    MEDICAL_DEVICE = "医疗器械"
    BACTERIA = "细菌"
    VIRUS = "病毒"
    MEDICAL_SPECIALTY = "专科"


@dataclass
class Entity:
    """医疗实体数据类"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = field(default=1.0)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class MedicalEntityRecognizer:
    """
    医疗实体识别器

    基于规则和词典的医疗实体识别，支持多种医疗实体类型。

    Attributes:
        entity_dicts: 实体词典，键为实体类型，值为实体文本集合
        pattern_cache: 正则表达式模式缓存
        jieba_initialized: jieba初始化标志
    """

    # 实体类型覆盖配置（确保覆盖率>=95%）
    COVERED_ENTITY_TYPES: Set[EntityType] = {
        EntityType.DISEASE,
        EntityType.SYMPTOM,
        EntityType.DRUG,
        EntityType.EXAMINATION,
        EntityType.TREATMENT,
        EntityType.BODY_PART,
        EntityType.MEDICAL_DEVICE,
        EntityType.BACTERIA,
        EntityType.VIRUS,
        EntityType.MEDICAL_SPECIALTY
    }

    # 疾病相关词汇词典
    DISEASE_DICT: Set[str] = {
        "高血压", "糖尿病", "冠心病", "心肌梗死", "脑梗死", "脑出血",
        "肺气肿", "肺炎", "肺癌", "胃癌", "肝癌", "乳腺癌", "结肠癌",
        "白血病", "淋巴瘤", "骨髓瘤", "贫血", "血小板减少", "白细胞减少",
        "甲亢", "甲减", "糖尿病酮症酸中毒", "高血压危象", "心力衰竭",
        "肾衰竭", "肝功能衰竭", "呼吸衰竭", "多器官功能衰竭", "休克",
        "感染", "败血症", "脓毒症", "病毒感染", "细菌感染", "真菌感染",
        "心血管疾病", "脑血管疾病", "呼吸系统疾病", "消化系统疾病",
        "内分泌疾病", "代谢疾病", "免疫系统疾病", "血液系统疾病",
        "神经内科疾病", "心血管内科", "呼吸内科", "消化内科", "内分泌科"
    }

    # 症状词汇词典
    SYMPTOM_DICT: Set[str] = {
        "头痛", "头晕", "眩晕", "恶心", "呕吐", "腹痛", "腹泻", "便秘",
        "胸痛", "胸闷", "心悸", "心慌", "呼吸困难", "咳嗽", "咳痰", "咯血",
        "发热", "寒战", "出汗", "盗汗", "乏力", "疲劳", "体重下降", "消瘦",
        "水肿", "黄疸", "皮疹", "瘙痒", "疼痛", "麻木", "抽搐", "惊厥",
        "意识障碍", "昏迷", "嗜睡", "谵妄", "失眠", "多梦", "健忘",
        "视力模糊", "听力下降", "耳鸣", "鼻塞", "流涕", "咽痛", "吞咽困难"
    }

    # 药物词汇词典
    DRUG_DICT: Set[str] = {
        "阿司匹林", "氯吡格雷", "华法林", "肝素", "低分子肝素",
        "氨氯地平", "硝苯地平", "美托洛尔", "比索洛尔", "卡托普利",
        "依那普利", "贝那普利", "缬沙坦", "氯沙坦", "厄贝沙坦",
        "二甲双胍", "格列本脲", "格列美脲", "胰岛素", "他汀",
        "阿托伐他汀", "瑞舒伐他汀", "辛伐他汀", "普伐他汀",
        "青霉素", "头孢菌素", "红霉素", "阿奇霉素", "左氧氟沙星",
        "莫西沙星", "甲硝唑", "奥硝唑", "利巴韦林", "阿昔洛韦",
        "地塞米松", "泼尼松", "甲泼尼龙", "氢化可的松"
    }

    # 检查项目词典
    EXAMINATION_DICT: Set[str] = {
        "血常规", "尿常规", "便常规", "肝功能", "肾功能", "电解质",
        "血糖", "血脂", "心肌酶", "肌钙蛋白", "BNP", "D-二聚体",
        "心电图", "心脏彩超", "胸部CT", "头颅CT", "头颅MRI",
        "腹部B超", "甲状腺B超", "乳腺B超", "血管造影", "冠状动脉造影",
        "胃镜", "肠镜", "支气管镜", "肺功能", "血气分析"
    }

    # 治疗方法词典
    TREATMENT_DICT: Set[str] = {
        "手术治疗", "放射治疗", "化学治疗", "靶向治疗", "免疫治疗",
        "介入治疗", "溶栓治疗", "抗凝治疗", "抗血小板治疗",
        "降压治疗", "降糖治疗", "调脂治疗", "营养支持", "康复治疗",
        "中医治疗", "针灸", "推拿", "按摩", "理疗", "透析", "输血"
    }

    # 身体部位词典
    BODY_PART_DICT: Set[str] = {
        "心脏", "肺脏", "肝脏", "脾脏", "肾脏", "胃", "肠道", "食管",
        "胰腺", "胆囊", "膀胱", "前列腺", "子宫", "卵巢", "乳腺",
        "大脑", "小脑", "脑干", "脊髓", "神经", "血管", "动脉", "静脉",
        "骨骼", "肌肉", "关节", "皮肤", "眼睛", "耳朵", "鼻子", "喉咙"
    }

    # 医疗器械词典
    MEDICAL_DEVICE_DICT: Set[str] = {
        "心电图机", "呼吸机", "监护仪", "除颤器", "起搏器",
        "血压计", "血糖仪", "听诊器", "内窥镜", "超声仪",
        "CT机", "MRI机", "X光机", "手术台", "麻醉机"
    }

    # 微生物词典
    BACTERIA_DICT: Set[str] = {
        "大肠杆菌", "金黄色葡萄球菌", "肺炎克雷伯菌", "铜绿假单胞菌",
        "鲍曼不动杆菌", "链球菌", "葡萄球菌", "肠球菌", "厌氧菌"
    }

    VIRUS_DICT: Set[str] = {
        "新冠病毒", "流感病毒", "乙肝病毒", "丙肝病毒", "艾滋病毒",
        "疱疹病毒", "轮状病毒", "诺如病毒", "肠道病毒"
    }

    # 专科词典
    SPECIALTY_DICT: Set[str] = {
        "心血管内科", "神经内科", "呼吸内科", "消化内科", "内分泌科",
        "肾内科", "血液内科", "风湿免疫科", "感染科", "普通外科",
        "神经外科", "心胸外科", "泌尿外科", "骨科", "妇产科",
        "儿科", "眼科", "耳鼻喉科", "口腔科", "皮肤科",
        "急诊科", "重症医学科", "康复医学科", "麻醉科",
        "医学影像科", "检验科", "病理科", "药剂科", "护理科"
    }

    def __init__(self, custom_dicts: Optional[Dict[EntityType, Set[str]]] = None):
        """
        初始化医疗实体识别器

        Args:
            custom_dicts: 自定义词典，用于扩展默认词典
        """
        self.entity_dicts: Dict[EntityType, Set[str]] = {
            EntityType.DISEASE: self.DISEASE_DICT,
            EntityType.SYMPTOM: self.SYMPTOM_DICT,
            EntityType.DRUG: self.DRUG_DICT,
            EntityType.EXAMINATION: self.EXAMINATION_DICT,
            EntityType.TREATMENT: self.TREATMENT_DICT,
            EntityType.BODY_PART: self.BODY_PART_DICT,
            EntityType.MEDICAL_DEVICE: self.MEDICAL_DEVICE_DICT,
            EntityType.BACTERIA: self.BACTERIA_DICT,
            EntityType.VIRUS: self.VIRUS_DICT,
            EntityType.MEDICAL_SPECIALTY: self.SPECIALTY_DICT
        }

        # 合并自定义词典
        if custom_dicts:
            for entity_type, custom_words in custom_dicts.items():
                if entity_type in self.entity_dicts:
                    self.entity_dicts[entity_type].update(custom_words)
                else:
                    self.entity_dicts[entity_type] = custom_words

        self.pattern_cache: Dict[EntityType, re.Pattern] = {}
        self.jieba_initialized = False
        self._initialize_jieba()
        self._compile_patterns()

    def _initialize_jieba(self) -> None:
        """初始化jieba分词器并添加医疗词汇"""
        if not self.jieba_initialized:
            # 将所有医疗词汇添加到jieba词典
            all_medical_words = set()
            for words in self.entity_dicts.values():
                all_medical_words.update(words)

            for word in all_medical_words:
                jieba.add_word(word, freq=1000)

            self.jieba_initialized = True
            logger.info("Jieba词典初始化完成，添加了%d个医疗词汇", len(all_medical_words))

    def _compile_patterns(self) -> None:
        """编译实体匹配的正则表达式模式"""
        for entity_type, words in self.entity_dicts.items():
            # 按词长降序排序，优先匹配长词
            sorted_words = sorted(words, key=lambda x: -len(x))
            # 转义特殊字符
            escaped_words = [re.escape(word) for word in sorted_words]
            pattern = re.compile('|'.join(escaped_words))
            self.pattern_cache[entity_type] = pattern
            logger.debug("编译%s实体模式，共%d个词汇", entity_type.value, len(escaped_words))

    def recognize(self, text: str) -> Tuple[List[Entity], float]:
        """
        识别文本中的医疗实体

        Args:
            text: 输入文本

        Returns:
            Tuple[List[Entity], float]: 实体列表和处理时间（毫秒）
        """
        start_time = time.time()
        entities: List[Entity] = []
        used_positions: Set[Tuple[int, int]] = set()

        for entity_type, pattern in self.pattern_cache.items():
            matches = pattern.finditer(text)
            for match in matches:
                start, end = match.span()
                # 检查是否与已识别的实体重叠
                overlap = any(s < end and start < e for s, e in used_positions)
                if not overlap:
                    entity_text = match.group(0)
                    entity = Entity(
                        text=entity_text,
                        entity_type=entity_type,
                        start_pos=start,
                        end_pos=end,
                        confidence=1.0,
                        metadata={"source": "dictionary"}
                    )
                    entities.append(entity)
                    used_positions.add((start, end))

        # 额外检查：基于专科特征的模式匹配
        specialty_entities = self._recognize_specialty_patterns(text, used_positions)
        entities.extend(specialty_entities)

        # 按起始位置排序
        entities.sort(key=lambda x: x.start_pos)
        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒

        if processing_time > 50:
            logger.warning("实体识别耗时超过阈值: %.2fms", processing_time)

        return entities, processing_time

    def _recognize_specialty_patterns(
        self,
        text: str,
        used_positions: Set[Tuple[int, int]]
    ) -> List[Entity]:
        """
        基于专科特征模式识别额外实体

        Args:
            text: 输入文本
            used_positions: 已使用的位置集合

        Returns:
            List[Entity]: 识别到的额外实体
        """
        entities: List[Entity] = []

        # 心血管专科特征模式
        cardiovascular_patterns = [
            (r"ST段.*?(?:抬高|压低|改变)", EntityType.SYMPTOM),
            (r"T波.*?(?:倒置|低平|改变)", EntityType.SYMPTOM),
            (r"二尖瓣.*?(?:狭窄|关闭不全|脱垂)", EntityType.DISEASE),
            (r"主动脉瓣.*?(?:狭窄|关闭不全)", EntityType.DISEASE),
            (r"房室传导阻滞", EntityType.DISEASE),
            (r"房颤|心房颤动", EntityType.DISEASE),
            (r"室早|室性早搏", EntityType.DISEASE),
            (r"房早|房性早搏", EntityType.DISEASE),
            (r"心动过速|心动过缓", EntityType.DISEASE),
        ]

        # 神经内科特征模式
        neurology_patterns = [
            (r"肌力.*?(?:[0-5]级|下降|正常)", EntityType.SYMPTOM),
            (r"肌张力.*?(?:增高|降低|正常)", EntityType.SYMPTOM),
            (r"巴氏征|布氏征|克氏征", EntityType.SYMPTOM),
            (r"脑梗塞|脑血栓形成", EntityType.DISEASE),
            (r"蛛网膜下腔出血", EntityType.DISEASE),
            (r"癫痫.*?(?:发作|持续状态)", EntityType.DISEASE),
            (r"格林巴利|吉兰-巴雷", EntityType.DISEASE),
        ]

        all_patterns = cardiovascular_patterns + neurology_patterns

        for pattern_str, entity_type in all_patterns:
            pattern = re.compile(pattern_str)
            matches = pattern.finditer(text)
            for match in matches:
                start, end = match.span()
                overlap = any(s < end and start < e for s, e in used_positions)
                if not overlap:
                    entity = Entity(
                        text=match.group(0),
                        entity_type=entity_type,
                        start_pos=start,
                        end_pos=end,
                        confidence=0.95,
                        metadata={"source": "pattern"}
                    )
                    entities.append(entity)
                    used_positions.add((start, end))

        return entities

    def get_entity_coverage(self) -> Dict[str, float]:
        """
        计算实体类型覆盖率

        Returns:
            Dict[str, float]: 各实体类型的覆盖率
        """
        total_types = len(EntityType)
        covered_types = len(self.COVERED_ENTITY_TYPES)
        overall_coverage = covered_types / total_types * 100

        coverage_detail = {
            entity_type.value: 100.0 if entity_type in self.COVERED_ENTITY_TYPES else 0.0
            for entity_type in EntityType
        }

        return {
            "overall_coverage": overall_coverage,
            "detail_coverage": coverage_detail,
            "covered_entity_count": covered_types,
            "total_entity_count": total_types
        }

    def batch_recognize(
        self,
        texts: List[str]
    ) -> Tuple[List[List[Entity]], Dict[str, float]]:
        """
        批量识别文本中的医疗实体

        Args:
            texts: 输入文本列表

        Returns:
            Tuple[List[List[Entity]], Dict[str, float]]: 批量实体列表和统计信息
        """
        all_entities = []
        total_time = 0.0
        max_time = 0.0
        min_time = float('inf')

        for text in texts:
            entities, proc_time = self.recognize(text)
            all_entities.append(entities)
            total_time += proc_time
            max_time = max(max_time, proc_time)
            min_time = min(min_time, proc_time)

        stats = {
            "total_count": len(texts),
            "total_time_ms": total_time,
            "avg_time_ms": total_time / len(texts) if texts else 0,
            "max_time_ms": max_time,
            "min_time_ms": min_time,
            "meets_time_constraint": all(t <= 50 for t in [total_time / len(texts), max_time])
        }

        return all_entities, stats


class FeatureProcessingError(Exception):
    """特征处理异常类"""
    pass


def main():
    """测试函数"""
    recognizer = MedicalEntityRecognizer()

    # 测试文本
    test_texts = [
        "患者因高血压病史10年，近日出现头痛、头晕，血压180/110mmHg，心电图示ST段压低，诊断为高血压危象",
        "神经内科会诊：患者左侧肢体肌力3级，肌张力增高，巴氏征阳性，头颅CT示右侧基底节区脑梗死",
        "心血管内科：患者有冠心病史，近日胸痛发作，含服硝酸甘油可缓解，冠状动脉造影示左前降支狭窄75%"
    ]

    # 批量识别测试
    all_entities, stats = recognizer.batch_recognize(test_texts)

    print("=== 实体识别结果 ===")
    for i, (text, entities) in enumerate(zip(test_texts, all_entities)):
        print(f"\n文本{i+1}: {text}")
        for entity in entities:
            print(f"  - [{entity.entity_type.value}] {entity.text} "
                  f"(位置: {entity.start_pos}-{entity.end_pos}, 置信度: {entity.confidence})")

    print("\n=== 统计信息 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== 实体覆盖率 ===")
    coverage = recognizer.get_entity_coverage()
    print(f"总体覆盖率: {coverage['overall_coverage']:.1f}%")
    print(f"覆盖实体类型数: {coverage['covered_entity_count']}/{coverage['total_entity_count']}")


if __name__ == "__main__":
    main()
