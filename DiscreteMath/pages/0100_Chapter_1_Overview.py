import streamlit as st

st.title("Chapter 1: Logic")

st.markdown("""
Logic provides a language for expressing statements precisely and reasoning about them correctly.

This chapter explores logic from several different perspectives:

- Propositions
- Conditional Statements
- Truth Tables
- Logical Equivalences
- Predicates
- Quantifiers
- Translation Between English and Logic
- Rules of Inference
""")

st.success("""
Suggested order:

Module 1 → Module 2 → Module 3 →
Module 4 → Module 5 → Module 6 →
Module 7 → Module 8 → Module 9
""")

st.header("Module Overview")

modules_df = {
    "Module": [
        "Module 1",
        "Module 2",
        "Module 3",
        "Module 4",
        "Module 5",
        "Module 6",
        "Module 7",
        "Module 8",
        "Module 9",
    ],
    "Topic": [
        "Propositions and Logical Operators",
        "Conditional Statements and Logical Equivalence",
        "Truth Tables and Logical Analysis",
        "Logical Equivalences and Laws",
        "Predicates and Quantifiers",
        "Negating Quantified Statements",
        "Nested Quantifiers",
        "Translating English Statements into Logic",
        "Rules of Inference",
    ],
}

st.table(modules_df)

st.info("""
By the end of this chapter, you will be able to:

• Determine whether a statement is a proposition.

• Build and analyze truth tables.

• Work with logical equivalences.

• Use predicates and quantifiers.

• Translate between English and logical notation.

• Apply valid rules of inference to draw conclusions.
""")