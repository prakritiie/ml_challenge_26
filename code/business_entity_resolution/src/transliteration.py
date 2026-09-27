"""V3 experiment: a self-contained, deterministic Devanagari -> Latin
phonetic transliteration prototype.

This is an ADDITIONAL representation used only by the V3 blocking
experiments (see blocking_v3_transliteration.py) -- it does NOT modify
normalization.py's real pipeline, and production blocking.py is untouched.
name_normalized / name_core / etc. on any record are never overwritten;
this module only ever produces new, separate translit_* values for records
that already contain Devanagari characters.

No external API, web lookup, business database, or geocoding is used
anywhere here -- every mapping below is a static, hand-authored character
table (Unicode block U+0900-U+097F), the same kind of fixed lookup already
used for LEGAL_SUFFIX_GROUPS / ADDRESS_ABBREVIATIONS in config.py.

Scope: Devanagari only (see README.md "V3 transliteration experiment" for
why it was chosen first -- it accounts for ~57% of true Indic-script
matches in both S2 and S3, more than 2x the next largest script).

Design / documented mapping decisions
--------------------------------------
- Consonants (vyanjana): each carries an inherent "a" sound (e.g. क = "ka").
  Standard IAST-ish simplification is used (aspirated consonants as two
  Latin letters: ख=kha, घ=gha, छ=chha, झ=jha, ठ=tha, ढ=dha, थ=tha, धdha,
  भ=bha, फ=pha). Retroflex vs. dental consonants (ट/त, ड/द, ण/न) are
  NOT distinguished (both -> t/d/n) -- a deliberate simplification since
  English spelling of Indian names doesn't distinguish them either, and it
  keeps the table small for this first prototype.
- Independent vowels (svara): used at the start of a word or after another
  vowel (e.g. अ=a, आ=aa, इ=i, ई=ii, उ=u, ऊ=uu, ऋ=ri, ए=e, ऐ=ai, ओ=o, औ=au).
- Matras (dependent vowel signs): attached to a consonant, REPLACE its
  inherent "a" (e.g. क+ि -> कि = "ki", not "kai"). Same vowel-sound mapping
  as the independent vowels, just applied post-consonant.
- Virama/halant (्): suppresses a consonant's inherent vowel entirely --
  used for consonant clusters (क्ष -> "ksh" not "kasha") and for a
  word-final consonant with no vowel sound.
- Anusvara (ं) / Chandrabindu (ँ): both nasalize the preceding vowel.
  Conservative choice: both map to "n" (not a context-sensitive m/n/ng
  choice before labial/velar consonants) -- simple and deterministic, at
  the cost of some phonetic precision (e.g. "सिंह" -> "sinha" rather than
  the more common English spelling "simha"/"singh"; that refinement is
  deferred, see README.md "next iteration").
- Visarga (ः): a trailing aspiration -> "h".
- Nukta consonants: Devanagari + nukta (़) represents sounds borrowed from
  Persian/Arabic/Dravidian (common in loanwords that appear in Indian
  business names, e.g. "ज़ोन" = "zone"). A small explicit table is used
  for the nukta forms actually observed to matter (ज़=z, फ़=f, क़=q,
  ख़=kh, ग़=g, ड़=r, ढ़=rh); any nukta not in this table falls back to
  its base consonant's plain sound.
- Devanagari digits (०-९) map to ASCII 0-9; danda/double-danda (।॥) map to
  "." (both rare in business names but included for completeness).
- Punctuation and spaces are passed through unchanged; any character not
  covered by a table above (e.g. an unexpected combining mark) is dropped
  rather than guessed at.
- **Schwa deletion policy: NONE, by design for this first prototype.**
  Modern spoken Hindi drops the final inherent "a" in most words (े.g.
  "लिमिटेड" is pronounced "limited", not "limiteda"), but implementing
  that correctly requires syllable-position rules that are easy to get
  wrong. Per the experiment's instructions, this prototype keeps every
  inherent vowel and measures the SIMPLE version first; the trailing-schwa
  mismatch this causes against English spellings is a known, documented
  limitation evaluated in blocking_v3_transliteration.py (Step 7).
"""

import re
from typing import Optional

DEVANAGARI_RANGE = (0x0900, 0x097F)
_DEVANAGARI_RE = re.compile(f"[{chr(DEVANAGARI_RANGE[0])}-{chr(DEVANAGARI_RANGE[1])}]")

# --- Nukta consonants (checked before plain consonants) ---------------------
NUKTA_CONSONANTS = {
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}

# --- Plain consonants (each carries an inherent "a") ------------------------
CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}

# --- Independent vowels (word-initial / post-vowel position) ---------------
INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ॠ": "rii", "ऌ": "li", "ॡ": "lii",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}

# --- Matras (dependent vowel signs; replace a consonant's inherent "a") ----
MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu",
    "ृ": "ri", "ॄ": "rii", "ॢ": "li", "ॣ": "lii",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}

VIRAMA = "्"          # ्  suppresses inherent vowel
ANUSVARA = "ं"        # ं
CHANDRABINDU = "ँ"    # ँ
VISARGA = "ः"         # ः
NASAL_MAP = {ANUSVARA: "n", CHANDRABINDU: "n"}

DEVANAGARI_DIGITS = {d: str(i) for i, d in enumerate("०१२३४५६७८९")}
DANDA = {"।": ".", "॥": "."}


def has_devanagari(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(_DEVANAGARI_RE.search(text))


def transliterate_devanagari(text: Optional[str]) -> Optional[str]:
    """Deterministic, conservative Devanagari -> Latin phonetic
    transliteration. No schwa deletion (see module docstring). Returns
    None for empty/None input; characters outside all the tables above
    (unexpected symbols) are silently dropped, not guessed at."""
    if not text:
        return None

    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        # Nukta consonant (2-codepoint sequence: base consonant + nukta,
        # OR a precomposed nukta character -- both forms appear in the data)
        two_char = ch + nxt
        if two_char in NUKTA_CONSONANTS:
            out.append(NUKTA_CONSONANTS[two_char])
            i += 2
            after = text[i] if i < n else ""
            if after == VIRAMA:
                i += 1
            elif after in MATRAS:
                out.append(MATRAS[after])
                i += 1
            else:
                out.append("a")
            continue

        if ch in NUKTA_CONSONANTS:
            out.append(NUKTA_CONSONANTS[ch])
            i += 1
            after = text[i] if i < n else ""
            if after == VIRAMA:
                i += 1
            elif after in MATRAS:
                out.append(MATRAS[after])
                i += 1
            else:
                out.append("a")
            continue

        if ch in CONSONANTS:
            out.append(CONSONANTS[ch])
            i += 1
            after = text[i] if i < n else ""
            if after == VIRAMA:
                i += 1  # cluster: no vowel between this consonant and the next
            elif after in MATRAS:
                out.append(MATRAS[after])
                i += 1
            else:
                out.append("a")  # inherent vowel, kept (no schwa deletion)
            continue

        if ch in INDEPENDENT_VOWELS:
            out.append(INDEPENDENT_VOWELS[ch])
            i += 1
            continue

        if ch in NASAL_MAP:
            out.append(NASAL_MAP[ch])
            i += 1
            continue

        if ch == VISARGA:
            out.append("h")
            i += 1
            continue

        if ch in DEVANAGARI_DIGITS:
            out.append(DEVANAGARI_DIGITS[ch])
            i += 1
            continue

        if ch in DANDA:
            out.append(DANDA[ch])
            i += 1
            continue

        if ch.isspace() or ch in ",.-/&'":
            out.append(ch)
            i += 1
            continue

        if ord(ch) < 128:  # pass through any already-Latin characters/digits
            out.append(ch)
            i += 1
            continue

        # Unrecognized character (stray combining mark, symbol, other
        # script mixed in, etc.) -- drop rather than guess.
        i += 1

    result = re.sub(r"\s+", " ", "".join(out)).strip()
    return result or None
