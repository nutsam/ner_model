import re
import spacy
from pprint import pprint
from collections import defaultdict
from ckip_transformers.nlp import CkipNerChunker

class TextPreprocessor:
    def __init__(self, stopwords=None, remove_punctuation=True):
        self.stopwords = stopwords or set()
        self.remove_punctuation = remove_punctuation
    
    def preprocess(self, text):
        """
        進行文本前處理：
        1. 移除網址
        2. 移除 HTML 標籤
        3. 移除特殊字元（可選是否保留標點符號）
        4. 統一空白
        5. 移除停用詞
        """
        text = re.sub(r'(http|ftp|https)://[^\s]+', '', text)  # 移除網址
        text = re.sub(r'<[^>]*>', '', text)  # 移除 HTML 標籤
        
        if self.remove_punctuation:
            text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s]', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()  # 統一空白
        
        if self.stopwords:
            text = ' '.join(word for word in text.split() if word not in self.stopwords)
        
        return text

class TextMasker:
    @staticmethod
    def replace_chinese_with_underscores(text):
        """將中文替換為相同長度的底線"""
        pattern = re.compile(r'[\u4e00-\u9fff0-9]+')
        return pattern.sub(lambda match: '_' * len(match.group(0)), text).replace('\n', '')

    @staticmethod
    def replace_english_with_underscores(text):
        """將英文單字替換為相同長度的底線"""
        pattern = re.compile(r'[A-Za-z]+')
        return pattern.sub(lambda match: '_' * len(match.group(0)), text).replace('\n', '')

class NamedEntityRecognizer:
    def __init__(self):
        self.ner_types = {"CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW",
                          "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON",
                          "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"}
        
        # 載入 spaCy 英文模型
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_en.add_pipe("gliner_spacy", config={"labels": list(self.ner_types)})
        
        # 載入 CKIP 中文模型
        self.nlp_zh = CkipNerChunker(model="bert-base")

    def recognize_english(self, text):
        doc = self.nlp_en(text)
        ner_dict = {ner: [] for ner in self.ner_types}
        
        for ent in doc.ents:
            cleaned_text = ent.text.strip().strip('_')
            if cleaned_text and ent.text == cleaned_text and ent.label_ in self.ner_types:
                ner_dict[ent.label_].append(cleaned_text)
        
        return ner_dict

    def recognize_chinese(self, text):
        result = self.nlp_zh([text])
        ner_dict = {ner: [] for ner in self.ner_types}
        
        for token in result[0]:
            cleaned_word = token.word.strip().strip('_')
            if cleaned_word and token.word == cleaned_word:
                ner_dict[token.ner].append(cleaned_word)
        
        return ner_dict

    def merge_results(self, ner_en_dict, ner_zh_dict):
        merged_dict = defaultdict(list)
        for d in [ner_en_dict, ner_zh_dict]:
            for key, value in d.items():
                merged_dict[key].extend(value)
        return dict(merged_dict)

if __name__ == "__main__":
    text = """
    昨天（2024/02/20），在東京Skytree舉辦了一場名為「AI Summit 2024」的科技論壇，吸引了來自 𝕊𝕀𝕃𝕀𝕔𝕆𝕟 𝕍𝕒𝕝𝕝𝕖𝕪 的專家參與。據報導，來自 OpenAI、DeepMind 和 𝔹𝕒𝕚𝕕𝕦 的工程師討論了人工智慧的未來趨勢，其中一項主題是關於 LLM 能否取代人類創造力？
    🌍 該活動由 NVIDIA 和 TSMC (台積電) 聯合贊助，並且吸引了包括 Microsoft 及 𝒜𝓅𝓅𝓁𝑒 在內的企業代表發表演講。來自新加坡的學者陳博士（Dr. Chen）提到：「AI 在 2024 年的發展將加速，但法規（GDPR & AI Act）仍需進一步完善。」
    📌 根據數據報告，2023 年 AI 產業的總值達到 $15.7B USD，同比增長 22.5%，其中 OpenAI 的 GPT 模型已擁有 超過 1.2 億 用戶。市場分析公司 Gartner 預測，2025 年 AI 市場將達到 $50B。
    💰 同時，日本政府宣布將投資 ¥100億 日圓於 AI 研究，並計劃在東京都內建立新的 AI 研究中心（Tokyo AI Research Center）。另外，中國的「天問二號」探測器計劃於 2026 年 登陸火星，這與 NASA 的 Artemis 計畫形成競爭。
    🔗 更多資訊請參考：https://www.aisummit2024.com/event?id=xyz_1234 或者發送 Email 至 info@aisummit.com 📧。
    """
    
    # 文本前處理
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.preprocess(text)
    
    # 文字遮罩
    english_masked = TextMasker.replace_chinese_with_underscores(cleaned_text)
    chinese_masked = TextMasker.replace_english_with_underscores(cleaned_text)
    
    # NER 識別
    ner = NamedEntityRecognizer()
    ner_en_results = ner.recognize_english(english_masked)
    ner_zh_results = ner.recognize_chinese(chinese_masked)
    merged_ner_results = ner.merge_results(ner_en_results, ner_zh_results)
    
    pprint(merged_ner_results)
