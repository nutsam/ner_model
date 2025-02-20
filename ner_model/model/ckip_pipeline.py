# model/ckip_pipeline.py
import logging
import re
from collections import defaultdict

import nltk
from flair.data import Sentence

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

from ner_model.model.model_util import CkipTransformerHandler
from ner_model.utils.text_cleaning import clean_chinese_text, clean_english_text

logger = logging.getLogger(__name__)


class CkipTransformerPipeline:
    """
    Production-ready pipeline for Named Entity Recognition using CKIP Transformer.
    Supports processing of Chinese, English, and mixed texts.
    """

    def __init__(
        self,
        device_ch,
        device_eng_ner,
        device_eng_pos,
        device_translate,
        batch_size_ch,
        batch_size_en,
        model_name,
        eng_ner_model,
        eng_pos_model,
        translate_model,
    ):
        self.batch_size_ch = batch_size_ch
        self.batch_size_en = batch_size_en
        self.device_ch = device_ch
        self.device_eng_ner = device_eng_ner
        self.device_eng_pos = device_eng_pos
        self.device_translate = device_translate

        self.model_name = model_name
        self.eng_ner_model = eng_ner_model
        self.eng_pos_model = eng_pos_model
        self.translate_model = translate_model

        self.english_stopwords = set(stopwords.words("english"))
        self.bert_model = (
            model_name if model_name in ["bert_tiny", "bert_base", "albert_base"] else "bert_tiny"
        )

        self._download_and_load_models()

    def _download_and_load_models(self):
        logger.info("Downloading and initializing models ...")
        CkipTransformerHandler.download_model_gdown(self.model_name)
        if self.model_name not in ["bert_tiny", "bert_base", "albert_base"]:
            CkipTransformerHandler.download_model_gdown(self.bert_model)
        CkipTransformerHandler.download_model_gdown(self.eng_ner_model)
        CkipTransformerHandler.download_model_gdown(self.eng_pos_model)
        # Load Chinese models
        self.ws_model = CkipTransformerHandler.load_model(
            self.bert_model, "ws", self.device_ch, self.batch_size_ch
        )
        self.pos_model = CkipTransformerHandler.load_model(
            self.bert_model, "pos", self.device_ch, self.batch_size_ch
        )
        self.ner_model = CkipTransformerHandler.load_model(
            self.model_name, "ner", self.device_ch, self.batch_size_ch
        )
        # Load English models
        self.eng_ner_driver = CkipTransformerHandler.load_model(
            self.eng_ner_model, "eng_ner", self.device_eng_ner, self.batch_size_en
        )
        self.eng_pos_driver = CkipTransformerHandler.load_model(
            self.eng_pos_model, "eng_pos", self.device_eng_pos, self.batch_size_en
        )
        logger.info("Model initialization complete.")

    def _classify_text(self, text):
        """根據英文比例將文本分成全英文句子與混合句子，並回傳清理後中文文本與兩類句子"""
        cleaned_ch = clean_chinese_text(text) or "-"
        cleaned_en = clean_english_text(text) or "-"
        sentences = [s for s in re.split(r"[，,。：:；;！!.？\?\n]", cleaned_en) if s]
        eng_sentences = [s for s in sentences if self._is_english_sentence(s)]
        mix_sentences = [
            s
            for s in sentences
            if not self._is_english_sentence(s) and re.findall(r"[A-Za-z0-9]{2,}", s)
        ]
        return cleaned_ch, eng_sentences, mix_sentences

    def _is_english_sentence(self, sentence):
        eng_parts = re.findall(r"[A-Za-z0-9]{2,}", sentence)
        if not sentence:
            return False
        ratio = (sum(len(s) for s in eng_parts) + max(len(eng_parts) - 1, 0)) / len(sentence)
        return ratio > 0.95

    def _prepare_model_inputs(self, texts):
        """對每筆文本進行分類，回傳清理後中文文本及英文／混合句子的 Sentence 物件與索引"""
        cleaned_texts, eng_inputs, mix_inputs = [], [], []
        eng_indices, mix_indices = [], []

        for idx, text in enumerate(texts):
            cleaned_ch, eng_sents, mix_sents = self._classify_text(text)
            cleaned_texts.append(cleaned_ch)
            if eng_sents:
                joined = ",".join(eng_sents) + "."
                eng_inputs.append(Sentence(joined[:512]))
                eng_indices.append(idx)
            if mix_sents:
                joined = ",".join(mix_sents) + "."
                mix_inputs.append(Sentence(joined[:512]))
                mix_indices.append(idx)
            else:
                mix_inputs.append(Sentence("-"))
                mix_indices.append(idx)
        return cleaned_texts, eng_inputs, eng_indices, mix_inputs, mix_indices

    def _predict_english(self, sentences, indices):
        """利用英文模型對句子進行預測，回傳根據原索引組織的斷詞、POS與NER結果"""
        num_texts = max(indices) + 1 if indices else 0
        ws_results = [[] for _ in range(num_texts)]
        pos_results = [[] for _ in range(num_texts)]
        ner_results = {i: {} for i in range(num_texts)}

        texts = [s.text[:512] for s in sentences]
        pos_preds = self.eng_pos_driver(texts)
        self.eng_ner_driver.predict(sentences, mini_batch_size=self.batch_size_en)

        for i, sent in enumerate(sentences):
            orig_idx = indices[i]
            temp_ws = [
                pos["word"].strip().upper()
                for pos in pos_preds[i]
                if pos["word"].strip() and pos["word"].strip().upper() in sent.text.upper()
            ]
            temp_pos = [
                pos["entity_group"]
                for pos in pos_preds[i]
                if pos["word"].strip() and pos["word"].strip().upper() in sent.text.upper()
            ]
            temp_ner = {}
            for span in sent.get_spans("ner"):
                tag = span.tag
                word = span.text.strip().upper()
                if word:
                    temp_ner.setdefault(tag, []).append(word)
            ws_results[orig_idx] = temp_ws
            pos_results[orig_idx] = temp_pos
            ner_results[orig_idx] = temp_ner
        return ws_results, pos_results, ner_results

    def _process_chinese(self, texts, max_length, use_delimiter, show_progress):
        """呼叫中文模型並過濾英文部份"""
        ws = self.ws_model(
            texts,
            batch_size=self.batch_size_ch,
            max_length=max_length,
            use_delim=use_delimiter,
            show_progress=show_progress,
        )
        pos = self.pos_model(
            ws,
            batch_size=self.batch_size_ch,
            max_length=max_length,
            use_delim=False,
            show_progress=show_progress,
        )
        ner = self.ner_model(
            texts,
            model_name=self.model_name,
            batch_size=self.batch_size_ch,
            max_length=max_length,
            use_delim=use_delimiter,
            show_progress=show_progress,
        )
        return self._filter_chinese(ws, pos, ner)

    def _filter_chinese(self, ws_list, pos_list, ner_list):
        """移除中文模型輸出中全英文的結果"""

        def is_english(token):
            token = token.replace(" ", "")
            return all((char.isalpha() or char.isdigit()) and char.isascii() for char in token)

        filtered_ws = [[token for token in tokens if not is_english(token)] for tokens in ws_list]
        filtered_pos = [
            [pos for token, pos in zip(tokens, pos_list[i]) if not is_english(token)]
            for i, tokens in enumerate(ws_list)
        ]
        filtered_ner = []
        for tokens in ner_list:
            filtered = [token for token in tokens if not is_english(token.word)]
            filtered_ner.append(filtered)
        return filtered_ws, filtered_pos, filtered_ner

    def run_pipeline(self, texts, max_length, use_delimiter, show_progress):
        (
            cleaned_texts,
            eng_inputs,
            eng_indices,
            mix_inputs,
            mix_indices,
        ) = self._prepare_model_inputs(texts)

        eng_ws, eng_pos, eng_ner = self._predict_english(eng_inputs, eng_indices)
        mix_ws, mix_pos, mix_ner = self._predict_english(mix_inputs, mix_indices)

        merged_ws = [e + m for e, m in zip(eng_ws, mix_ws)]
        merged_pos = [e + m for e, m in zip(eng_pos, mix_pos)]
        merged_ner = {i: {**eng_ner.get(i, {}), **mix_ner.get(i, {})} for i in range(len(texts))}

        ch_ws, ch_pos, ch_ner = self._process_chinese(
            cleaned_texts, max_length, use_delimiter, show_progress
        )
        return merged_ws, merged_pos, merged_ner, ch_ws, ch_pos, ch_ner, cleaned_texts

    def _extract_entities(self, ws_tokens, pos_tokens, ner_tokens, max_len=35):
        """從斷詞與 POS 結果及 NER 輸出中抽取實體並過濾短詞"""
        entities = defaultdict(set)
        for token, pos in zip(ws_tokens, pos_tokens):
            token, pos = token.strip(), pos.strip()
            if len(token) > 1 and len(pos) <= 4:
                if "A" in pos:
                    entities["adj"].add(token)
                if "D" in pos:
                    entities["adv"].add(token)
                if "V" in pos:
                    entities["v"].add(token)
                if pos in ["Na", "Nb", "Nc", "PRON"]:
                    entities["n"].add(token)
        for ent in ner_tokens:
            word = ent.word.strip().upper()
            label = ent.ner
            valid = ("BIO" in self.model_name and label in {"PER", "LOC", "ORG"}) or (
                label
                in {"PERSON", "ORG", "EVENT", "FAC", "PRODUCT", "GPE", "LOC", "NORP", "WORK_OF_ART"}
            )
            if valid and word:
                entities[label.lower()].add(word.replace(" ", "&nbsp;"))
        for key, words in entities.items():
            entities[key] = {w for w in words if len(w.replace("&nbsp;", "")) <= max_len}
        return entities

    def process_output(
        self, original_texts, eng_ws, eng_pos, eng_ner, ch_ws, ch_pos, ch_ner, cleaned_texts
    ):
        """整合英文與中文的實體結果"""
        results = {}
        for idx in range(len(original_texts)):
            eng_entities = self._extract_entities(eng_ws[idx], eng_pos[idx], [])
            ch_entities = self._extract_entities(ch_ws[idx], ch_pos[idx], ch_ner[idx])
            combined = {**eng_entities}
            for label, words in ch_entities.items():
                combined[label] = combined.get(label, set()) | words
            results[idx] = combined
        return results

    def update_entities(self, ner_results):
        """將完整名稱整合到名詞中，避免部分斷詞衝突"""
        targets = ["n", "adj"]
        for entities in ner_results.values():
            for label, words in list(entities.items()):
                if label not in ["n", "v", "adv", "neu", "fw", "adj", "det"]:
                    for full in list(words):
                        parts = full.split("&nbsp;")
                        if len(parts) > 1:
                            for target in targets:
                                if target in entities:
                                    entities[target] -= set(parts)
                                    entities[target].add(full)
        return ner_results

    def enhanced_cleaning(self, ner_results, original_texts):
        """依據自訂規則進一步清理 NER 結果"""

        def should_keep(token):
            token = token.strip()
            if token.startswith("'") or token.startswith("-"):
                return False
            if re.search(r"[,/\\]", token):
                return False
            has_chinese = bool(re.search(r"[\u4e00-\u9fff]", token))
            if has_chinese and any(ch.isdigit() for ch in token) and len(token) < 5:
                return False
            if " " in token and len(token) < 5:
                return False
            if (
                has_chinese
                and any(ch.isalpha() for ch in token)
                and any(ch.isdigit() for ch in token)
                and " " in token
                and len(token) < 8
            ):
                return False
            return True

        def normalize(token):
            token = token.replace("\n", "").replace("&nbsp;", " ").strip()
            return token.replace(" ", "&nbsp;")

        cleaned = {}
        for idx, entities in ner_results.items():
            cleaned[idx] = {
                label: {normalize(word) for word in words if should_keep(word)}
                for label, words in entities.items()
                if words
            }
        return cleaned

    def get_named_entities(self, texts, use_batch, max_length, use_delimiter, show_progress):
        if use_batch:
            eng_ws, eng_pos, eng_ner, ch_ws, ch_pos, ch_ner, cleaned_texts = self.run_pipeline(
                texts, max_length, use_delimiter, show_progress
            )
            raw_results = self.process_output(
                texts, eng_ws, eng_pos, eng_ner, ch_ws, ch_pos, ch_ner, cleaned_texts
            )
        else:
            raw_results = {}
            for idx, line in enumerate(texts):
                eng_ws, eng_pos, eng_ner, ch_ws, ch_pos, ch_ner, cleaned_texts = self.run_pipeline(
                    [line], max_length, use_delimiter, show_progress
                )
                raw_results[idx] = self.process_output(
                    [line], eng_ws, eng_pos, eng_ner, ch_ws, ch_pos, ch_ner, cleaned_texts
                )
        updated = self.update_entities(raw_results)
        return self.enhanced_cleaning(updated, texts)
