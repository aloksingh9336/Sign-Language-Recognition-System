import difflib


# ============================================================
# COMMON ENGLISH WORDS
# ============================================================

WORD_LIST = [
    "A",
    "ABOUT",
    "ALL",
    "AM",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BECAUSE",
    "CAN",
    "COME",
    "COULD",
    "DAY",
    "DO",
    "DOING",
    "DON'T",
    "FOR",
    "FROM",
    "GET",
    "GO",
    "GOOD",
    "HAVE",
    "HE",
    "HELLO",
    "HELP",
    "HER",
    "HERE",
    "HIM",
    "HIS",
    "HOW",
    "I",
    "IF",
    "IN",
    "IS",
    "IT",
    "KNOW",
    "LIKE",
    "ME",
    "MY",
    "NEED",
    "NO",
    "NOT",
    "NOW",
    "OF",
    "ON",
    "ONE",
    "OR",
    "OUR",
    "PLEASE",
    "SEE",
    "SHE",
    "SO",
    "SORRY",
    "THANK",
    "THANKS",
    "THAT",
    "THE",
    "THEIR",
    "THERE",
    "THEY",
    "THIS",
    "TO",
    "TODAY",
    "TOMORROW",
    "UP",
    "US",
    "WANT",
    "WAS",
    "WE",
    "WELCOME",
    "WHAT",
    "WHEN",
    "WHERE",
    "WHO",
    "WHY",
    "WILL",
    "WITH",
    "YES",
    "YOU",
    "YOUR",

    # Common sign-language-related words
    "SIGN",
    "LANGUAGE",
    "SCHOOL",
    "COLLEGE",
    "STUDENT",
    "TEACHER",
    "FRIEND",
    "FAMILY",
    "HOME",
    "WORK",
    "FOOD",
    "WATER",
    "MOTHER",
    "FATHER",
    "BROTHER",
    "SISTER",
    "NAME",
    "GOOD",
    "MORNING",
    "NIGHT",
    "AFTERNOON",
    "WELCOME",
    "SORRY",
    "PLEASE",
    "THANK",
    "THANKS",
    "LOVE",
    "HAPPY",
    "SAD",
    "HELP",
]


# ============================================================
# REMOVE DUPLICATES
# ============================================================

WORD_LIST = sorted(
    list(
        set(
            WORD_LIST
        )
    )
)


# ============================================================
# AUTOCOMPLETE
# ============================================================

def get_autocomplete_suggestions(
    current_word,
    max_suggestions=3
):

    """
    Returns words that start with the
    currently typed letters.
    """

    if not current_word:

        return []


    current_word = (
        current_word
        .strip()
        .upper()
    )


    suggestions = []


    for word in WORD_LIST:

        if word.startswith(
            current_word
        ):

            if word != current_word:

                suggestions.append(
                    word
                )


    return suggestions[
        :max_suggestions
    ]


# ============================================================
# WORD CORRECTION
# ============================================================

def get_word_corrections(
    current_word,
    max_suggestions=3
):

    """
    Returns similar words when the
    current word contains spelling errors.
    """

    if not current_word:

        return []


    current_word = (
        current_word
        .strip()
        .upper()
    )


    matches = difflib.get_close_matches(

        current_word,

        WORD_LIST,

        n=max_suggestions,

        cutoff=0.55

    )


    return matches


# ============================================================
# COMBINED SUGGESTIONS
# ============================================================

def get_suggestions(
    current_word,
    max_suggestions=3
):

    """
    First tries autocomplete.

    If autocomplete does not find enough
    results, fuzzy spelling correction
    is used.
    """

    if not current_word:

        return []


    current_word = (
        current_word
        .strip()
        .upper()
    )


    # ========================================================
    # AUTOCOMPLETE
    # ========================================================

    autocomplete = (
        get_autocomplete_suggestions(

            current_word,

            max_suggestions

        )
    )


    # ========================================================
    # FUZZY CORRECTION
    # ========================================================

    corrections = (
        get_word_corrections(

            current_word,

            max_suggestions

        )
    )


    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    suggestions = []


    for word in autocomplete:

        if word not in suggestions:

            suggestions.append(
                word
            )


    for word in corrections:

        if word not in suggestions:

            suggestions.append(
                word
            )


    return suggestions[
        :max_suggestions
    ]


# ============================================================
# BEST WORD CORRECTION
# ============================================================

def correct_word(
    current_word
):

    """
    Returns the closest matching word.

    If no suitable match is found,
    the original word is returned.
    """

    if not current_word:

        return ""


    current_word = (
        current_word
        .strip()
        .upper()
    )


    matches = get_word_corrections(

        current_word,

        max_suggestions=1

    )


    if matches:

        return matches[0]


    return current_word


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "WORD CORRECTION TEST"
    )

    print("=" * 60)


    test_words = [

        "HEL",

        "HELO",

        "HELP",

        "THAN",

        "THNKS",

        "WELC"

    ]


    for word in test_words:

        print()

        print(
            "Input:",
            word
        )


        suggestions = (
            get_suggestions(
                word
            )
        )


        print(
            "Suggestions:",
            suggestions
        )


        corrected = (
            correct_word(
                word
            )
        )


        print(
            "Best correction:",
            corrected
        )


    print()

    print("=" * 60)

    print(
        "TEST COMPLETED"
    )

    print("=" * 60)