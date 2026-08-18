import streamlit as st
import pandas as pd

from Chapter_1_utils import *

LAW_INFO = {

    "Idempotent Laws": {
        "left1": "p ∨ p",
        "right1": "p",
        "left2": "p ∧ p",
        "right2": "p",
        "explanation":
            "Repeating the same proposition does not change its truth value."
    },

    "Associative Laws": {
        "left1": "(p ∨ q) ∨ r",
        "right1": "p ∨ (q ∨ r)",
        "left2": "(p ∧ q) ∧ r",
        "right2": "p ∧ (q ∧ r)",
        "explanation":
            "Changing the grouping does not change the result."
    },

    "Commutative Laws": {
        "left1": "p ∨ q",
        "right1": "q ∨ p",
        "left2": "p ∧ q",
        "right2": "q ∧ p",
        "explanation":
            "Changing the order does not change the result."
    },

    "Distributive Laws": {
        "left1": "p ∨ (q ∧ r)",
        "right1": "(p ∨ q) ∧ (p ∨ r)",
        "left2": "p ∧ (q ∨ r)",
        "right2": "(p ∧ q) ∨ (p ∧ r)",
        "explanation":
            "OR distributes over AND, and AND distributes over OR."
    },

    "Identity Laws": {
        "left1": "p ∨ F",
        "right1": "p",
        "left2": "p ∧ T",
        "right2": "p",
        "explanation":
            "Combining a proposition with False using OR or True using AND does not change the proposition."
    },

    "Domination Laws": {
        "left1": "p ∧ F",
        "right1": "F",
        "left2": "p ∨ T",
        "right2": "T",
        "explanation":
            "Some truth values dominate the result."
    },

    "Double Negation Law": {
        "left1": "¬¬p",
        "right1": "p",
        "left2": "",
        "right2": "",
        "explanation":
            "Applying NOT twice restores the original proposition."
    },

    "Complement Laws": {
        "left1": "p ∧ ¬p",
        "right1": "F",
        "left2": "p ∨ ¬p",
        "right2": "T",
        "explanation":
            "A proposition and its negation cannot both be true, and one of them must be true."
    },

    "DeMorgan's Laws": {
        "left1": "¬(p ∨ q)",
        "right1": "¬p ∧ ¬q",
        "left2": "¬(p ∧ q)",
        "right2": "¬p ∨ ¬q",
        "explanation":
            "NOT can be pushed inside parentheses by changing OR to AND and AND to OR."
    },

    "Absorption Laws": {
        "left1": "p ∨ (p ∧ q)",
        "right1": "p",
        "left2": "p ∧ (p ∨ q)",
        "right2": "p",
        "explanation":
            "The extra expression is absorbed by p."
    },

    "Conditional Identities": {
        "left1": "p → q",
        "right1": "¬p ∨ q",
        "left2": "p ↔ q",
        "right2": "(p → q) ∧ (q → p)",
        "explanation":
            "Conditional and biconditional statements can be rewritten using simpler operators."
    }
}

st.title("Module 4: Logical Equivalences and Laws")

st.markdown("""
In Module 3, we used truth tables to determine whether two propositions are logically equivalent.

In this module, we study important logical equivalence laws that can be used to simplify and analyze logical propositions.
""")

st.subheader("Logical Equivalence Laws")

st.markdown("""
A logical equivalence law states that two propositions have the same truth value for every assignment of truth values to their variables.

Equivalence laws allow us to rewrite one proposition as another logically equivalent proposition.
""")

laws_df = pd.DataFrame(
    {
        "Law": [
            "Idempotent Laws",
            "Associative Laws",
            "Commutative Laws",
            "Distributive Laws",
            "Identity Laws",
            "Domination Laws",
            "Double Negation Law",
            "Complement Laws",
            "DeMorgan's Laws",
            "Absorption Laws",
            "Conditional Identities",
        ],
        "": [
            "p ∨ p ≡ p",
            "(p ∨ q) ∨ r ≡ p ∨ (q ∨ r)",
            "p ∨ q ≡ q ∨ p",
            "p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r)",
            "p ∨ F ≡ p",
            "p ∧ F ≡ F",
            "¬¬p ≡ p",
            """p ∧ ¬p ≡ F\n\n¬T ≡ F""",
            "¬(p ∨ q) ≡ ¬p ∧ ¬q",
            "p ∨ (p ∧ q) ≡ p",
            "p → q ≡ ¬p ∨ q",
        ],
        " ": [
            "p ∧ p ≡ p",
            "(p ∧ q) ∧ r ≡ p ∧ (q ∧ r)",
            "p ∧ q ≡ q ∧ p",
            "p ∧ (q ∨ r) ≡ (p ∧ q) ∨ (p ∧ r)",
            "p ∧ T ≡ p",
            "p ∨ T ≡ T",
            "",
            """p ∨ ¬p ≡ T\n\n¬F ≡ T""",
            "¬(p ∧ q) ≡ ¬p ∨ ¬q",
            "p ∧ (p ∨ q) ≡ p",
            "p ↔ q ≡ (p → q) ∧ (q → p)",
        ],
    }
)

st.table(laws_df)

st.subheader("Explore a Law")

selected_law = st.selectbox(
    "Choose a law",
    [
        "Idempotent Laws",
        "Associative Laws",
        "Commutative Laws",
        "Distributive Laws",
        "Identity Laws",
        "Domination Laws",
        "Double Negation Law",
        "Complement Laws",
        "DeMorgan's Laws",
        "Absorption Laws",
        "Conditional Identities",
    ]
)

info = LAW_INFO[selected_law]

st.markdown(f"**{selected_law}**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Form 1**")
    st.code(
        f"{info['left1']}   ≡   {info['right1']}",
        language="text"
    )

with col2:
    if info["left2"]:
        st.markdown("**Form 2**")
        st.code(
            f"{info['left2']}   ≡   {info['right2']}",
            language="text"
        )

st.info(info["explanation"])

st.subheader("Verify Using Truth Tables")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        f"**{info['left1']} ≡ {info['right1']}**"
    )

    truth_df_1 = generate_equivalence_truth_table(
        info["left1"],
        info["right1"]
    )

    st.dataframe(
        truth_df_1,
        hide_index=True,
        width="stretch"
    )

    st.success("""
    The final columns are identical.

    Therefore the two expressions are logically equivalent.
    """)

with col2:

    if info["left2"]:

        st.markdown(
            f"**{info['left2']} ≡ {info['right2']}**"
        )

        truth_df_2 = generate_equivalence_truth_table(
            info["left2"],
            info["right2"]
        )

        st.dataframe(
            truth_df_2,
            hide_index=True,
            width="stretch"
        )

        st.success("""
        The final columns are identical.

        Therefore the two expressions are logically equivalent.
        """)

with st.expander("Additional Programming Connection: Short-Circuit Evaluation"):

    st.markdown("""
Many programming languages use **short-circuit evaluation**
for Boolean expressions.

For example:
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
    ```python
    if True or p:
        do_something()
    ```

    Since the left side is already True, the value of p does not affect the result. 
    The expression p is never evaluated.
        """)

    with col2:
        st.markdown("""
```python
if False and p:
    do_something()
```

Since the left side is already False, the value of p does not affect the result. 
The expression p is never evaluated.
        """)

    st.markdown("""
However, short-circuit evaluation depends on the order of the operands.
The evaluator does not rewrite expressions using commutative logical equivalence laws.

For example:
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
```python
if p or True:
    do_something()
```
        """)

    with col2:
        st.markdown("""
```python
if p and False:
    do_something()
```
        """)

    st.markdown("""
Even though

p ∨ True ≡ True ∨ p

and

p ∧ False ≡ False ∧ p

the evaluator processes the left operand first.
""")

    st.markdown("""In other words,
    """)


    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
    May Avoid Evaluating p
    """)

        st.markdown("""
    ```text
    True ∨ p
    False ∧ p
    ```
    """)

    with col2:

        st.markdown("""
    Must Evaluate p First
    """)

        st.markdown("""
    ```text
    p ∨ True
    p ∧ False
    ```
    """)

    st.markdown("""
This distinction can matter. Consider
```python 
counter = 0 

def increment(): 
    global counter 
    counter += 1 
    return True
```
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
    In

    ```python 
    True or increment()
    ```

    increment() is never called.

    The value of counter remains unchanged.
""")

    with col2:
        st.markdown("""
    In

    ```python 
    increment() or True
    ```

    increment() is called first.

    The value of counter changes. """)
