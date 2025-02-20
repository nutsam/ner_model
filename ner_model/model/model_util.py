# model_util.py
import os
import logging
import zipfile
import torch
import gdown
import flair
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flair.models import SequenceTagger
from ner_model.model.process.ckiplab_models.driver import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

logger = logging.getLogger(__name__)

MODEL_IDS = {
    'bert_tiny': '1O-nQrlKSRtLmWR3U_yJ4azaoiPXPAy1_',
    'bert_base': '1N-VZGjBKKC2tLlLE5TOJzp_hsE0i9bka',
    'albert_base': '1XobDKSWBNRcUuOR0G7cFyvmUVBVVszMK',
    'finetune_pink_onenote5': '15Kp-ENg5Hb1T7UpMVDCNojAyLPbAMWhk',
    'BIO_finetune_pink_msra': '15inE6qk2F08mMv_7vHoOG8Zrp0y8cEFy',
    'eng_ontonotes': '1793ARrQQT1RHvdx1bNDmPz2NdCnkoiMI',
    'eng_ontonotes_large': '1QDsUD8xEYIeDNDzOImn6AOxOzM2WlC1v',
    'eng_upos': '1ciXHDd1g1nZWBhD6fy6-JUwFVEdJyhjD',
    'eng_vblagoje_pos': '1Uu5H4Avle0NVM5cuX9tCiE7KXg3ttkMb',
    'translate': '1thDmhhgdSCyrrOViD0LYTWobzSyw2Rri'
}

class CkipTransformerHandler:
    """
    Utility class for handling CKIP Transformer models.
    Provides functionality for downloading and loading CKIP Transformer models for:
      - Word segmentation (ws)
      - Part-of-speech tagging (pos)
      - Named entity recognition (ner)
      - 以及英文模型與翻譯模型
    """

    @staticmethod
    def download_model_gdown(model_name: str, root: str = '~/.ckip_transformer_models') -> None:
        """
        Downloads a CKIP Transformer model via gdown and extracts it.
        
        Args:
            model_name (str): Name of the model.
            root (str): Root directory to store downloaded models.
        """
        drive_url = f'https://drive.google.com/uc?id={MODEL_IDS[model_name]}'
        root_dir = os.path.expanduser(root)
        output_dir = os.path.join(root_dir, f'model_{model_name}')

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if os.path.exists(output_dir):
            logger.info("Model '%s' already downloaded at %s", model_name, output_dir)
            return

        zip_path = os.path.join(root_dir, 'models.zip')
        logger.info("Downloading model '%s' ...", model_name)
        gdown.download(drive_url, zip_path, quiet=False)

        logger.info("Extracting model '%s' ...", model_name)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # 排除 __MACOSX 資料夾
            file_list = [f for f in zip_ref.namelist() if "__MACOSX" not in f]
            zip_ref.extractall(os.path.dirname(output_dir), members=file_list)

        os.remove(zip_path)
        logger.info("Model '%s' downloaded and extracted to %s", model_name, output_dir)

    @staticmethod
    def load_model(model_name: str, model_type: str, device_type: int, batch_size: int,
                   root: str = '~/.ckip_transformer_models'):
        """
        Loads a CKIP Transformer model driver based on model type.
        
        Args:
            model_name (str): Name of the model.
            model_type (str): Type of model ('ws', 'pos', 'ner', 'eng_ner', 'eng_pos', 'translate').
            device_type (int): Device to use (-1 for CPU, >=0 for GPU index).
            batch_size (int): Batch size for processing.
            root (str): Root directory where models are stored.
        
        Returns:
            A model driver instance.
        """
        root_dir = os.path.join(os.path.expanduser(root), f'model_{model_name}')
        model_path = os.path.join(root_dir, "pytorch_model.bin")

        # 驗證模型檔案是否存在
        if not (os.path.exists(os.path.join(root_dir, model_type)) or os.path.exists(model_path)):
            logger.error("Model file for '%s' type '%s' does not exist in %s", model_name, model_type, root_dir)
            return None

        logger.info("Loading model '%s' of type '%s' ...", model_name, model_type)

        # 依照不同 model_type 載入相應驅動
        if model_type in ['ws', 'pos', 'ner']:
            driver = CkipTransformerHandler._load_chinese_model(model_name, model_type, device_type, root_dir)
        elif model_type in ['eng_ner', 'eng_pos']:
            driver = CkipTransformerHandler._load_english_model(model_name, model_type, device_type, batch_size, model_path, root_dir)
        elif model_name == 'translate' and model_type == 'translate':
            driver = CkipTransformerHandler._load_translation_model(model_name, device_type, batch_size, model_path, root_dir)
        else:
            logger.error("Unsupported model type: %s", model_type)
            driver = None

        return driver

    @staticmethod
    def _load_chinese_model(model_name: str, model_type: str, device_type: int, root_dir: str):
        model_dir = os.path.join(root_dir, model_type)
        if model_type == 'ws':
            return CkipWordSegmenter(model_name=model_dir, device=device_type)
        if model_type == 'pos':
            return CkipPosTagger(model_name=model_dir, device=device_type)
        if model_type == 'ner':
            return CkipNerChunker(model_name=model_dir, device=device_type)
        return None

    @staticmethod
    def _load_english_model(model_name: str, model_type: str, device_type: int, batch_size: int, model_path: str, root_dir: str):
        # 設定裝置
        if torch.cuda.is_available() and device_type != -1:
            flair.device = torch.device(f'cuda:{device_type}')
        else:
            flair.device = torch.device('cpu')

        # 若使用 eng_vblagoje_pos 則使用 transformers pipeline
        if model_type == 'eng_pos' and model_name == 'eng_vblagoje_pos':
            pipeline_device = -1 if device_type == -1 else device_type
            return pipeline(
                "token-classification",
                model=root_dir,
                tokenizer=root_dir,
                device=pipeline_device,
                aggregation_strategy="simple",
                batch_size=batch_size
            )
        # 其他英文模型透過 Flair 直接載入
        return SequenceTagger.load(model_path)

    @staticmethod
    def _load_translation_model(model_name: str, device_type: int, batch_size: int, model_path: str, root_dir: str):
        device_str = "cpu" if device_type == -1 else f"cuda:{device_type}"
        pipeline_device = -1 if device_type == -1 else device_type
        tokenizer = AutoTokenizer.from_pretrained(root_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(root_dir).to(device_str)
        return pipeline("translation", model=model, tokenizer=tokenizer, device=pipeline_device, batch_size=batch_size)
