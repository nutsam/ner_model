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
        📢 Breaking News | 重大消息！ 🎉

        📅 昨天（𝟮𝟬𝟮𝟱/𝟬𝟮/𝟮𝟬），台灣🇹🇼總統 李安然 (An-Ran Li) 在台北101 🏙 舉行記者會，宣布政府將投資 NT$1️⃣5️⃣0️⃣ 億 用於 AI 發展計畫。該計畫由 行政院科技部 (MOST) 負責，預計於 𝟮𝟬𝟮𝟲 年開始實施。

        🌏 國際企業大戰 AI 領域！ 在矽谷 🌁，Meta、Google DeepMind、𝕺𝖕𝖊𝖓𝕬𝖎 和 阿里巴巴 宣布建立「𝓐𝓘 𝓡𝓮𝓼𝓮𝓪𝓻𝓬𝓱 𝓒𝓸𝓵𝓵𝓪𝓫𝓸𝓻𝓪𝓽𝓲𝓸𝓷」，共同研究大型語言模型 (LLM) 的可解釋性與公平性。同時，日本🇯🇵的 SONY 也宣布投入 ¥500 億 用於 AI 晶片開發。

        🎮 2025 年電競界盛事！ 今年 7 月，英雄聯盟 (LoL) 世界大賽 (Worlds 2025) 將於 美國🏆洛杉磯 Crypto.com Arena 舉辦，來自 北美 (LCS)、歐洲 (LEC)、韓國 (LCK) 和中國 (LPL) 的戰隊將爭奪冠軍🏅。根據數據分析，去年 LCK 總收入達 $75.3M，而 LPL 則高達 ¥5.8B。

        💸 全球市場趨勢 根據 Bloomberg 的最新報告：

        2024 年全球 AI 產業估值達 $𝟮𝟎𝟬B USD 💰，同比增長 35.2% 📈。
        𝒕𝒉𝒆 S&P 500 指數突破 5,𝟬𝟬𝟬 點，創下歷史新高🏦！
        2025 年，亞馬遜 (Amazon) 預計投入 €10B 於雲端 AI 計畫 ☁️。
        🎭 文化與娛樂 Netflix 近日推出了一部新劇《𝕯𝖎𝖌𝖎𝖙𝖆𝖑 𝕯𝖗𝖊𝖆𝖒𝖘》，該劇講述了一名 AI 工程師在虛擬世界與現實之間穿梭的故事📽。評論家認為：「這部劇展示了 21 世紀數位化與 AI 技術的未來。」

        💻 詭異的 AI 錯誤？ 一位 Twitter 用戶 @ai_error💥 指出，ChatGPT 於 𝟮𝟬𝟮𝟰 年 生成了一篇不存在的新聞，導致大量網友誤以為日本政府倒閉 (假的！)。這也讓 OpenAI 宣布將強化 AI 內容真實性驗證。

        📢 聯絡我們 想了解更多詳情？請訪問：

        🌍 https://www.aitechnews.com/?ref=test🔗
        📧 contact@ai-news.com
        ☎️ +1-(800)-555-📞-AI24
    """
    
    # 文本前處理
    preprocessor = TextPreprocessor(remove_punctuation=False)
    cleaned_text = preprocessor.preprocess(text)
    print(cleaned_text)
    print('- '* 50)
    
    # 文字遮罩
    english_masked = TextMasker.replace_chinese_with_underscores(cleaned_text)
    chinese_masked = TextMasker.replace_english_with_underscores(cleaned_text)
    
    print(english_masked)
    print('- '* 50)
    print(chinese_masked)
    
    # NER 識別
    ner = NamedEntityRecognizer()
    ner_en_results = ner.recognize_english(english_masked)
    ner_zh_results = ner.recognize_chinese(chinese_masked)
    merged_ner_results = ner.merge_results(ner_en_results, ner_zh_results)
    
    pprint(merged_ner_results)
