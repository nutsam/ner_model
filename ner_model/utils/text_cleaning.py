# utils/text_cleaning.py
import re

# 正則：匹配 URL
URL_REGEX = re.compile(
    r"\s*((http|https)://|www\.)((?!http://|https://|www\.)[A-Za-z0-9+/=:/?#[\]!$&'()*+,;.%\-_~]|%[0-9a-fA-F]{2})+\s*"
)


def clean_chinese_text(text: str) -> str:
    """
    清理中文文本：移除 URL、emoji 與多餘符號。
    """
    text = URL_REGEX.sub(" ", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)
    # 清理重複標點與其他特殊符號
    for pattern in [r"—{2,}", r"-{2,}", r"-={2,}", r"一{2,}", r"&{2,}", r"\.{2,}", r"\+{2,}"]:
        text = re.sub(pattern, " ", text)
    for char in [
        " - ",
        "\xa0",
        ";",
        "\t",
        ")",
        "(",
        "*",
        "%",
        "～",
        "±",
        ":",
        "/",
        "<",
        ">",
        "©",
        "=",
        "★",
        "②",
        "→",
        "：",
        "https://.",
        "https",
        "http",
    ]:
        text = text.replace(char, " ")
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    return text


def clean_english_text(text: str) -> str:
    """
    清理英文文本：移除 URL、emoji 與多餘符號。
    """
    text = URL_REGEX.sub(" ", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)
    for pattern in [r"—{2,}", r"-{2,}", r"-={2,}", r"一{2,}", r"&{2,}", r"\+{2,}", r"\.{2,}"]:
        text = re.sub(pattern, " ", text)
    for char in [
        " - ",
        "\xa0",
        ";",
        "\t",
        ")",
        "(",
        "*",
        "%",
        "～",
        "±",
        ":",
        "/",
        "<",
        ">",
        "©",
        "=",
        "★",
        "②",
        "→",
        "：",
        "《",
        "》",
        "【",
        "】",
        "（",
        "）",
        "https://.",
        "https",
        "http",
    ]:
        text = text.replace(char, " ")
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    return text
