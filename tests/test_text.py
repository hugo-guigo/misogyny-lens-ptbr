from app.text import analyze, clean


def test_clean_keeps_length_and_offsets():
    text = "Olha @fulana isso: https://x.com/abc 123 MULHER!!"
    out = clean(text)
    assert len(out) == len(text)
    assert "@" not in out and "http" not in out and "123" not in out
    assert out[text.index("MULHER"):text.index("MULHER") + 6] == "mulher"


def test_tokens_point_to_original_text():
    text = "As Mulheres não deveriam dirigir 🚗"
    for tok in analyze(text):
        assert text[tok.start:tok.end] == tok.text


def test_filters_stopwords_and_short_lemmas():
    lemmas = [t.lemma for t in analyze("de o a em mulher") if t.lemma]
    assert lemmas == ["mulher"]
