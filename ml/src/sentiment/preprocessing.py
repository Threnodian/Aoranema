import re


def clean_text(text: str) -> str:
    """
    Preprocessing minimal untuk input sentiment IndoBERT.

    Tidak melakukan stemming, stopword removal,
    atau menghapus tanda baca karena model Transformer
    membutuhkan konteks teks sebanyak mungkin.
    """

    if not isinstance(text, str):
        raise TypeError("Text harus berupa string.")

    # Hilangkan whitespace di awal/akhir
    text = text.strip()

    # Gabungkan whitespace berlebih
    text = re.sub(r"\s+", " ", text)

    if not text:
        raise ValueError("Text tidak boleh kosong.")

    return text