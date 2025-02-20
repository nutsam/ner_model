#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
This module implements the CKIP Transformers NLP drivers.
"""

__author__ = "Mu Yang <http://muyang.pro>"
__copyright__ = "2023 CKIP Lab"
__license__ = "GPL-3.0"

from typing import (
    List,
)

import numpy as np

from .util import (
    CkipTokenClassification,
    NerToken,
)

################################################################################################################################


class CkipWordSegmenter(CkipTokenClassification):
    """The word segmentation driver.

    Parameters
    ----------
        model : ``str`` *optional*, defaults to "bert-base".
            The pretrained model name provided by CKIP Transformers.
        model_name : ``str`` *optional*, overwrites **model**
            The custom pretrained model name (e.g. ``'ckiplab/bert-base-chinese-ws'``).
        device : ``int`` or ``torch.device``, *optional*, defaults to -1
            Device ordinal for CPU/GPU supports.
            Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id.
    """

    _model_names = {
        "albert-tiny": "ckiplab/albert-tiny-chinese-ws",
        "albert-base": "ckiplab/albert-base-chinese-ws",
        "bert-tiny": "ckiplab/bert-tiny-chinese-ws",
        "bert-base": "ckiplab/bert-base-chinese-ws",
    }

    def __init__(
        self,
        model: str = "bert-base",
        **kwargs,
    ):
        model_name = kwargs.pop("model_name", self._get_model_name(model))
        super().__init__(model_name=model_name, **kwargs)

    def __call__(
        self,
        input_text: List[str],
        *,
        use_delim: bool = False,
        **kwargs,
    ) -> List[List[str]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[str]``
                The input sentences. Each sentence is a string.
            use_delim : ``bool``, *optional*, defaults to False
                Segment sentence (internally) using ``delim_set``.
            delim_set : `str`, *optional*, defaults to ``'，,。：:；;！!？?'``
                Used for sentence segmentation if ``use_delim=True``.
            batch_size : ``int``, *optional*, defaults to 256
                The size of mini-batch.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).
            show_progress : ``int``, *optional*, defaults to True
                Show progress bar.
            pin_memory : ``bool``, *optional*, defaults to True
                Pin memory in order to accelerate the speed of data transfer to the GPU. This option is incompatible with
                multiprocessing. Disabled on CPU device.

        Returns
        -------
            ``List[List[str]]``
                A list of list of words (``str``).
        """

        # Call model
        (
            logits,
            index_map,
        ) = super().__call__(input_text, use_delim=use_delim, **kwargs)

        # Post-process results
        output_text = []
        for sent_data in zip(input_text, index_map):
            output_sent = []
            word = ""
            for input_char, logits_index in zip(*sent_data):
                if logits_index is None:
                    if word:
                        output_sent.append(word)
                    output_sent.append(input_char)
                    word = ""
                else:
                    logits_b, logits_i = logits[logits_index]

                    if logits_b > logits_i:
                        if word:
                            output_sent.append(word)
                        word = input_char
                    else:
                        word += input_char

            if word:
                output_sent.append(word)
            output_text.append(output_sent)

        return output_text


################################################################################################################################


class CkipPosTagger(CkipTokenClassification):
    """The part-of-speech tagging driver.

    Parameters
    ----------
        model : ``str`` *optional*, defaults to "bert-base".
            The pretrained model name provided by CKIP Transformers.
        model_name : ``str`` *optional*, overwrites **model**
            The custom pretrained model name (e.g. ``'ckiplab/bert-base-chinese-pos'``).
        device : ``int`` or ``torch.device``, *optional*, defaults to -1
            Device ordinal for CPU/GPU supports.
            Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id.
    """

    _model_names = {
        "albert-tiny": "ckiplab/albert-tiny-chinese-pos",
        "albert-base": "ckiplab/albert-base-chinese-pos",
        "bert-tiny": "ckiplab/bert-tiny-chinese-pos",
        "bert-base": "ckiplab/bert-base-chinese-pos",
    }

    def __init__(
        self,
        model: str = "bert-base",
        **kwargs,
    ):
        model_name = kwargs.pop("model_name", self._get_model_name(model))
        super().__init__(model_name=model_name, **kwargs)

    def __call__(
        self,
        input_text: List[List[str]],
        *,
        use_delim: bool = True,
        **kwargs,
    ) -> List[List[str]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[List[str]]``
                The input sentences. Each sentence is a list of strings (words).
            use_delim : ``bool``, *optional*, defaults to True
                Segment sentence (internally) using ``delim_set``.
            delim_set : `str`, *optional*, defaults to ``'，,。：:；;！!？?'``
                Used for sentence segmentation if ``use_delim=True``.
            batch_size : ``int``, *optional*, defaults to 256
                The size of mini-batch.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).
            show_progress : ``int``, *optional*, defaults to True
                Show progress bar.
            pin_memory : ``bool``, *optional*, defaults to True
                Pin memory in order to accelerate the speed of data transfer to the GPU. This option is incompatible with
                multiprocessing. Disabled on CPU device.

        Returns
        -------
            ``List[List[str]]``
                A list of list of POS tags (``str``).
        """

        # Call model
        (
            logits,
            index_map,
        ) = super().__call__(input_text, use_delim=use_delim, **kwargs)

        # Get labels
        id2label = self.model.config.id2label

        # Post-process results
        output_text = []
        for sent_data in zip(input_text, index_map):
            output_sent = []
            for input_char, logits_index in zip(*sent_data):
                if logits_index is None or input_char.isspace():
                    label = "WHITESPACE"
                else:
                    label = id2label[np.argmax(logits[logits_index])]
                output_sent.append(label)
            output_text.append(output_sent)

        return output_text


################################################################################################################################


class CkipNerChunker(CkipTokenClassification):
    """The named-entity recognition driver.

    Parameters
    ----------
        model : ``str`` *optional*, defaults to "bert-base".
            The pretrained model name provided by CKIP Transformers.
        model_name : ``str`` *optional*, overwrites **model**
            The custom pretrained model name (e.g. ``'ckiplab/bert-base-chinese-ner'``).
        device : ``int`` or ``torch.device``, *optional*, defaults to -1
            Device ordinal for CPU/GPU supports.
            Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id.
    """

    _model_names = {
        "albert-tiny": "ckiplab/albert-tiny-chinese-ner",
        "albert-base": "ckiplab/albert-base-chinese-ner",
        "bert-tiny": "ckiplab/bert-tiny-chinese-ner",
        "bert-base": "ckiplab/bert-base-chinese-ner",
    }

    def __init__(
        self,
        model: str = "bert-base",
        **kwargs,
    ):
        model_name = kwargs.pop("model_name", self._get_model_name(model))
        super().__init__(model_name=model_name, **kwargs)

    def __call__(
        self,
        input_text: List[str],
        *,
        use_delim: bool = False,
        model_name: str = "",
        **kwargs,
    ) -> List[List[NerToken]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[str]``
                The input sentences. Each sentence is a string.
            model_name : `str`, defaults to ""
            use_delim : ``bool``, *optional*, defaults to False
                Segment sentence (internally) using ``delim_set``.
            delim_set : `str`, *optional*, defaults to ``'，,。：:；;！!？?'``
                Used for sentence segmentation if ``use_delim=True``.
            batch_size : ``int``, *optional*, defaults to 256
                The size of mini-batch.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).
            show_progress : ``int``, *optional*, defaults to True
                Show progress bar.
            pin_memory : ``bool``, *optional*, defaults to True
                Pin memory in order to accelerate the speed of data transfer to the GPU. This option is incompatible with
                multiprocessing. Disabled on CPU device.

        Returns
        -------
            ``List[List[NerToken]]``
                A list of list of entities (:class:`~.util.NerToken`).
        """

        # Call model     
        (
            logits,
            index_map,
        ) = super().__call__(input_text, use_delim=use_delim, **kwargs)

        # Get labels
        id2label = self.model.config.id2label
        if_BIOES = any(label.split('-')[0] == 'E' for label in list(id2label.values()))

        # Post-process results
        output_text = []
        if if_BIOES == False:
            output_text.append(self.finetune_model_of_different_label_postprocess_logic(input_text, logits, index_map, id2label, model_name))
        else:
            for sent_data in zip(input_text, index_map):
                output_sent = []
                entity_word = None
                entity_ner = None
                entity_idx0 = None
                for index_char, (
                    input_char,
                    logits_index,
                ) in enumerate(zip(*sent_data)):
                    if logits_index is None:
                        label = "O"
                    else:
                        label = id2label[np.argmax(logits[logits_index])]

                    if label == "O":
                        entity_ner = None
                        continue

                    bioes, ner = label.split("-")

                    if bioes == "S":
                        output_sent.append(
                            NerToken(
                                word=input_char,
                                ner=ner,
                                idx=(
                                    index_char,
                                    index_char + len(input_char),
                                ),
                            )
                        )
                        entity_ner = None
                    elif bioes == "B":
                        entity_word = input_char
                        entity_ner = ner
                        entity_idx0 = index_char
                    elif bioes == "I":
                        if entity_ner == ner:
                            entity_word += input_char
                        else:
                            entity_ner = None
                    elif bioes == "E":
                        if entity_ner == ner:
                            entity_word += input_char
                            output_sent.append(
                                NerToken(
                                    word=entity_word,
                                    ner=entity_ner,
                                    idx=(
                                        entity_idx0,
                                        index_char + len(input_char),
                                    ),
                                )
                            )
                        entity_ner = None
                    

                output_text.append(output_sent)

        return output_text


    def finetune_model_of_different_label_postprocess_logic(self, input_text, logits, index_map, id2label, model_name):
        """Call the driver for finetune model of different label method (BIO).

        Parameters
        ----------
            input_text : ``List[str]``
                The input sentences. Each sentence is a string.
            logits : ``numpy.ndarray``
                The logits array containing model predictions.
            index_map : ``List[List[Tuple[int, int]]]``
                A list representing the mapping of indices to their corresponding positions in the logits array.
            id2label : ``dict``
                A dictionary mapping numeric labels to their corresponding named entity labels.
            model_name : `str`
                The name of the user-selected model.

        Returns
        -------
            ``output_sent : List[NerToken]``
                A list of NerToken objects representing named entities in a sentence.
        """
        for sent_data in zip(input_text, index_map):
            output_sent = []
            entity_word = None
            entity_ner = None
            entity_idx0 = None
            record_label = "first-word"
            for index_char, (
                input_char,
                logits_index,
            ) in enumerate(zip(*sent_data)):
                if logits_index is None:
                    label = "O"
                    logits_index_record = None
                else:
                    logits_index_record = 1
                    label = id2label[np.argmax(logits[logits_index])]

                if label == "O" and record_label.split("-")[0] == "I" and logits_index_record is not None:
                    output_sent.append(
                            NerToken(
                                word=entity_word,
                                ner=entity_ner,
                                idx=(
                                    entity_idx0,
                                    index_char + (len(input_char)-1),
                                ),
                            )
                        )
                    record_label = label
                    entity_ner = None
                    continue


                if len(label.split("-")) == 2:
                    bioes, ner = label.split("-")
                    if bioes == "B":
                        entity_word = input_char
                        entity_ner = ner
                        entity_idx0 = index_char
                    elif bioes == "I":
                        if entity_ner == ner:
                            entity_word += input_char
                        else:
                            entity_ner = None
                    record_label = label
            return output_sent