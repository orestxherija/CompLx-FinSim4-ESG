import re
import typing


def remove_final_period(doc: str) -> str:
    """
    Remove final period character from string
    """
    return doc[:-1] if doc[-1] == "." else doc


def lowercase_text(doc: str) -> str:
    """
    Convert text to lowercase
    """
    return doc.lower()


def add_space_around_string(doc: str, string_list: typing.List[str]) -> str:
    """
    Add a space character around each string in a list of strings
    """
    match = re.compile("(" + "|\\".join(string_list) + ")")
    return match.sub(r" \1 ", doc)


def add_space_around_comma(doc: str) -> str:
    """
    Add a space around commas, unless they are digit separators
    """
    n = len(doc)
    new_doc = ""
    for i, c in enumerate(doc):
        if c != ",":
            new_doc += c
        else:
            if i == 0 or i == n - 1:
                new_doc += ""
            else:
                if doc[i-1].isdigit() and doc[i+1].isdigit():
                    new_doc += c
                else:
                    new_doc += " " + c + " "
    return new_doc


def transform_doc_with_mapper(doc: str, mapper: typing.Dict[str, str]) -> str:
    """
    Replace substrings in a document according to a predetermined mapping
    """
    for k in mapper:
        match = re.compile(k)
        doc = match.sub(mapper[k], doc)
    return doc


def label_str_to_int(text):
    """
    Convert `str` labels to `int`
    """
    if text == "sustainable":
        return 1
    elif text == "unsustainable":
        return 0
    else:
        raise f"Label `{text}` is not recognized."


special_characters = [
    "“",
    "”",
    ":",
    "(",
    ")",
    "[",
    "]",
    ";",
    "?",
    "‘",
    "’",
    ">",
    "\"",
    "€",
    "$"
]

possessive_chars = [
    "’"  # multiple special cases here
]

character_map = {
    "–": "-",
    "§": "",
    "*": "",
    "®": "",
    "—": "-",
    "‑": "-",
    "‐": "-",
}

string_map = {
    "\xad": "-",
    "Climate Action 100+": "CA100+",
    " & ": " and ",
    " - ": " ",
    "°C": " °C ",
}
