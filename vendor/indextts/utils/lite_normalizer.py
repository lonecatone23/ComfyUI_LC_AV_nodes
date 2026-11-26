"""LC AV: a small English text normalizer that stands in for wetext (WeTextProcessing FSTs) in IndexTTS-2's frontend.

Written for LC AV (not upstream code). It spells out what wetext's English "tn" grammar spells out most often, in the
same words: cardinals ("one hundred and thirty nine"), years ("nineteen eighty seven", "twenty twenty five", "twelve
hundred"), decimals ("three point one four"), negatives, ordinals (1st, 22nd), thousands separators, percent, money
($ GBP EUR, "twelve point five dollars"), clock times ("ten thirty PM", 8:00 -> "eight"), simple fractions (1/2),
leading zeros ("oh oh seven"), decades ("the nineties"), "&" and a few titles (Mr., Mrs., Ms., Dr.).
Not covered (wetext did these): dates like 12/25/2020, units (km/h, kg), phone numbers, URLs / e-mail, roman numerals,
and all Chinese normalization (Chinese numbers stay as digits). Text the lite rules don't touch passes through as is.
"""

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}
_DIGIT = {str(i): _ONES[i] for i in range(10)}
_FRAC = {2: ("half", "halves"), 4: ("quarter", "quarters")}
_TITLES = {"Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss", "Dr.": "doctor"}
_CUR = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"), "€": ("euro", "euros")}


def _two(n):
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + (" " + _ONES[o] if o else "")


def _three(n):
    h, r = divmod(n, 100)
    if not h:
        return _two(r)
    return _ONES[h] + " hundred" + (" and " + _two(r) if r else "")


def cardinal(n):
    n = int(n)
    if n < 0:
        return "negative " + cardinal(-n)
    if n < 1000:
        return _three(n)
    parts = []
    for size, name in _SCALES:
        if n >= size:
            q, n = divmod(n, size)
            parts.append(cardinal(q) + " " + name)
    if n:
        parts.append(_three(n))
    return " ".join(parts)


def year(n):
    if n % 1000 == 0 or 2000 < n < 2010:
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return _two(hi) + " hundred"
    return _two(hi) + " " + (("oh " + _ONES[lo]) if lo < 10 else _two(lo))


def ordinal(n):
    words = cardinal(n).split(" ")
    last = words[-1]
    if last in _ORD:
        words[-1] = _ORD[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _number(s, as_year=True):
    """digits (maybe with , separators and a .decimal part) -> words"""
    neg = s.startswith("-")
    s = s.lstrip("-")
    whole, _, frac = s.partition(".")
    whole = whole.replace(",", "")
    if len(whole) > 1 and whole.startswith("0") and not frac:
        out = " ".join("oh" if d == "0" else _DIGIT[d] for d in whole)
    elif as_year and not frac and "," not in s and len(whole) == 4 and 1000 <= int(whole) <= 2099:
        out = year(int(whole))
    else:
        out = cardinal(int(whole or "0"))
    if frac:
        out += " point " + " ".join(_DIGIT[d] for d in frac)
    return ("negative " + out) if neg else out


_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"


def normalize(text):
    for k, v in _TITLES.items():
        text = re.sub(r"(?<![\w.])" + re.escape(k), v, text)
    text = re.sub(r"\s&\s", " and ", text)

    def money(m):
        one, many = _CUR[m.group(1)]
        amount = m.group(2)
        if "." in amount:
            amount = amount.rstrip("0").rstrip(".")  # $12.50 -> twelve point five dollars, as wetext reads it
        word = one if amount == "1" else many
        return _number(amount, as_year=False) + " " + word
    text = re.sub(r"([$£€])\s?(" + _NUM + r")", money, text)

    def clock(m):
        h, mi = int(m.group(1)), int(m.group(2))
        out = cardinal(h) if mi == 0 else cardinal(h) + " " + ("oh " + _ONES[mi] if mi < 10 else _two(mi))
        return out + (" " + m.group(3).replace(".", "").upper() if m.group(3) else "")
    text = re.sub(r"\b(\d{1,2}):([0-5]\d)\b(?:\s?([AaPp]\.?[Mm]\.?)(?!\w))?", clock, text)

    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), text)
    def decade(m):
        w = _number(m.group(1)).split(" ")
        w[-1] = w[-1][:-1] + "ies" if w[-1].endswith("y") else w[-1] + "s"
        return " ".join(w)
    text = re.sub(r"\b(\d{1,3}0)s\b", decade, text)

    def frac(m):
        a, b = int(m.group(1)), int(m.group(2))
        if b in _FRAC:
            name = _FRAC[b][0] if a == 1 else _FRAC[b][1]
        else:
            name = ordinal(b) + ("" if a == 1 else "s")
        return cardinal(a) + " " + name
    text = re.sub(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])", frac, text)

    text = re.sub(r"(" + _NUM + r")\s?%", lambda m: _number(m.group(1), as_year=False) + " percent", text)
    text = re.sub(r"(?<![\w.])(-?)(" + _NUM + r")(?![\d])",
                  lambda m: _number(m.group(1) + m.group(2)), text)
    return text


class LiteEnglishNormalizer:
    def normalize(self, text):
        return normalize(text)


class DigitsOnlyChineseNormalizer:
    """Chinese normalization is not available without wetext: Chinese text passes through unchanged. Text with no
    Chinese characters (IndexTTS routes digit-only text here) still gets the English number reading."""

    _cjk = re.compile(r"[一-鿿]")

    def normalize(self, text):
        return text if self._cjk.search(text) else normalize(text)
