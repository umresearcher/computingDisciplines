import streamlit as st
import pandas as pd

st.title("Module 8: Translating English Statements into Logic")

st.markdown("""
In Modules 5–7, we learned how predicates, quantifiers, and nested quantifiers are used to express logical statements.

In this module, we learn how to translate statements written in English into logical notation.
""")

st.subheader("Why Translation Matters")

st.markdown("""
Many statements in mathematics, computer science, and databases are written in English.

Translating these statements into logical notation helps us analyze them precisely.
""")

st.subheader("Example")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
**English Statement**

```text
Every student submitted the assignment.
```
""")

with col2:
    st.markdown("""
**Logical Form**

```text
∀student Submitted(student)
```
""")

st.subheader("Common Translation Patterns")

patterns_df = pd.DataFrame(
    {
        "English Phrase": [
            "Every ...",
            "There exists ...",
            "No ...",
            "Not every ...",
            "Every ... has a ...",
            "There exists a ... for every ...",
        ],
        "Logical Form": [
            "∀",
            "∃",
            "¬∃",
            "¬∀",
            "∀x ∃y",
            "∃y ∀x",
        ],
    }
)

st.dataframe(
    patterns_df,
    hide_index=True,
    width="content",
)

st.subheader("Examples with Predicates")

examples_df = pd.DataFrame(
    {
        "English Statement": [
            "Every student is enrolled in a course.",
            "Some course has a student enrolled.",
            "No student is enrolled in MTH 328.",
            "Not every student submitted the assignment.",
            "There exists a student who did not submit the assignment.",
            "There exists a course in which every student is enrolled.",
            "Every course has at least one enrolled student.",
        ],
        "Possible Logical Form": [
            "∀student ∃course Enrolled(student, course)",
            "∃course ∃student Enrolled(student, course)",
            "¬∃student Enrolled(student, MTH 328)",
            "¬∀student Submitted(student)",
            "∃student ¬Submitted(student)",
            "∃course ∀student Enrolled(student, course)",
            "∀course ∃student Enrolled(student, course)",
        ],
    }
)

col1, col2 = st.columns([2,5])

with col1:
    st.markdown("")
    st.markdown("""
    Assume the following predicate definitions:

    ```text
    Submitted(student) is True when
    the student submitted the assignment.
    
    Enrolled(student, course) is True when
    the student is enrolled in the course. 
    ```
    """)

with col2:
    st.dataframe(
        examples_df,
        hide_index=True,
        width="stretch",
    )

translations = {

    "Every student submitted the assignment.":
        "∀student Submitted(student)",

    "There exists a student who submitted the assignment.":
        "∃student Submitted(student)",

    "No student submitted the assignment.":
        "¬∃student Submitted(student)",

    "Not every student submitted the assignment.":
        "¬∀student Submitted(student)",

    "There exists a student who did not submit the assignment.":
        "∃student ¬Submitted(student)",

    "Every student is enrolled in a course.":
        "∀student ∃course Enrolled(student, course)",

    "There exists a course in which every student is enrolled.":
        "∃course ∀student Enrolled(student, course)",

    "Every course has at least one enrolled student.":
        "∀course ∃student Enrolled(student, course)",
}

reverse_translations = {
    v: k
    for k, v in translations.items()
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("English to Logic")

    statement = st.selectbox(
        "Choose an English statement",
        list(translations.keys())
    )

    st.markdown("**Logical Form**")

    st.code(
        translations[statement],
        language="text"
    )

with col2:
    st.subheader("Logic to English")

    logical_form = st.selectbox(
        "Choose a logical statement",
        list(reverse_translations.keys())
    )

    st.markdown("**English Translation**")

    st.code(
        reverse_translations[logical_form],
        language="text"
    )

col1, col2 = st.columns([5,2])

with col1:
    st.subheader("Common Translation Pitfalls")

    pitfalls_df = pd.DataFrame(
        {
            "English Statement": [
                "No student submitted the assignment.",
                "Not every student submitted the assignment.",
                "There exists a student who did not submit the assignment.",
            ],
            "Logical Form": [
                "¬∃student Submitted(student)",
                "¬∀student Submitted(student)",
                "∃student ¬Submitted(student)",
            ],
        }
    )

    st.dataframe(
        pitfalls_df,
        hide_index=True,
        width="stretch",
    )

    st.info("""
    These statements have different meanings.

    "No student ..." means that nobody satisfies the predicate.

    "Not every ..." means that at least one element does not satisfy the predicate.

    "There exists ... who did not ..." means there is a specific counterexample.
    """)

with col2:
    st.subheader("Quick Translation Reference")

    translation_df = pd.DataFrame( { 
        "English Phrase": [ "Every", "For all", "There exists", "Some", "At least one", "No", ], 
        "Logical Symbol": [ "∀", "∀", "∃", "∃", "∃", "¬∃", ], 
        } ) 

    st.markdown("")
    st.dataframe( translation_df, hide_index=True, width="content", )

