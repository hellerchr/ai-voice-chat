from conversation import Sentences


def split(*chunks):
    sentences = Sentences()
    out = [s for chunk in chunks for s in sentences.add(chunk)]
    return out + sentences.finish()


def test_sentences_are_cut_as_they_stream():
    sentences = Sentences()
    assert sentences.add("Zwei plus zwei ist vier") == []
    assert sentences.add(". Noch ") == ["Zwei plus zwei ist vier."]
    assert sentences.finish() == ["Noch"]


def test_numbers_and_abbreviations_stay_inside_their_sentence():
    assert split("Am 3. Oktober kostet es 3.50 Euro, z. B. bei Dr. Weber. Gut!") == [
        "Am 3. Oktober kostet es 3.50 Euro, z. B. bei Dr. Weber.", "Gut!"]


def test_a_long_sentence_is_split_only_at_a_comma():
    long = "Igel fressen Käfer, Würmer und Schnecken " * 6
    first = split(long)[0]
    assert first.endswith(",") and len(first) <= 200
