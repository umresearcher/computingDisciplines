import streamlit as st
import pandas as pd

from Chapter_1_utils import *

st.title("Module 9: Rules of Inference")

st.markdown("""
In the previous modules, we learned how to represent logical statements.

In this module, we learn how to use logical statements to draw valid conclusions.
""")

st.subheader("Drawing Conclusions")

st.markdown("""
Consider the premises

```text
If it is raining, then the sidewalk is wet, and
It is raining.
```

What conclusion can we draw? """)

st.success("""
Conclusion:

```text
The sidewalk is wet.
```
""")

st.subheader("Modus Ponens")

st.markdown("""
The reasoning above uses the rule

```text
p → q
p
-----
q
```

This rule is called Modus Ponens. """)

st.subheader("Why Is Modus Ponens Valid?") 

df1 = generate_truth_table("p → q")
df2 = generate_truth_table("(p → q) ∧ p")
df3 = generate_truth_table("((p → q) ∧ p) → q")

display_df = pd.DataFrame()

display_df["p"] = df1["p"]
display_df["q"] = df1["q"]
display_df["p → q"] = df1["Result"]
display_df["(p → q) ∧ p"] = df2["Result"]
display_df["((p → q) ∧ p) → q"] = df3["Result"]

col1, col2 = st.columns([1,3])

with col1:
    st.markdown(""" Consider the proposition 

    ```text 
    ((p → q) ∧ p) → q
    ```
    If this proposition is always True, then Modus Ponens is a valid rule of inference. """)

with col2:
    st.dataframe( display_df, hide_index=True, width="stretch" )

classification = classify_expression(df3)

if classification == "Tautology":
    st.success("""
The proposition is a tautology. Therefore Modus Ponens is a valid rule of inference.
""")

st.info("""
A valid rule of inference guarantees that whenever all premises are True, the conclusion is also True.
""")

st.subheader("Common Rules of Inference")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("**Modus Ponens**")

    st.code(
        """p
p → q
-----
q""",
        language="text"
    )

with col2:

    st.markdown("**Addition**")

    st.code(
        """p
-----
p ∨ q""",
        language="text"
    )

with col3:

    st.markdown("**Conjunction**")

    st.code(
        """p
q
-----
p ∧ q""",
        language="text"
    )

with col4:

    st.markdown("**Disjunctive Syllogism**")

    st.code(
        """p ∨ q
¬p
-----
q""",
        language="text"
    )

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("**Modus Tollens**")

    st.code(
        """¬q
p → q
-----
¬p""",
        language="text"
    )

with col2:

    st.markdown("**Simplification**")

    st.code(
        """p ∧ q
-----
p""",
        language="text"
    )

with col3:

    st.markdown("**Hypothetical Syllogism**")

    st.code(
        """p → q
q → r
-----
p → r""",
        language="text"
    )

with col4:

    st.markdown("**Resolution**")

    st.code(
        """p ∨ q
¬p ∨ r
---------
q ∨ r""",
        language="text"
    )

st.subheader("Explore a Rule")

RULE_INFO = {

    "Modus Ponens": {
        "premises": [
            "p",
            "p → q",
        ],
        "premises_expr": "(p → q) ∧ p",
        "conclusion": "q",
        "tautology": "((p → q) ∧ p) → q",
    },

    "Modus Tollens": {
        "premises": [
            "¬q",
            "p → q",
        ],
        "premises_expr": "(p → q) ∧ ¬q",
        "conclusion": "¬p",
        "tautology": "((p → q) ∧ ¬q) → ¬p",
    },

    "Addition": {
        "premises": [
            "p",
        ],
        "premises_expr": "p",
        "conclusion": "p ∨ q",
        "tautology": "p → (p ∨ q)",
    },

    "Simplification": {
        "premises": [
            "p ∧ q",
        ],
        "premises_expr": "p ∧ q",
        "conclusion": "p",
        "tautology": "(p ∧ q) → p",
    },

    "Conjunction": {
        "premises": [
            "p",
            "q",
        ],
        "premises_expr": "p ∧ q",
        "conclusion": "p ∧ q",
        "tautology": "(p ∧ q) → (p ∧ q)",
    },

    "Hypothetical Syllogism": {
        "premises": [
            "p → q",
            "q → r",
        ],
        "premises_expr": "(p → q) ∧ (q → r)",
        "conclusion": "p → r",
        "tautology": "((p → q) ∧ (q → r)) → (p → r)",
    },

    "Disjunctive Syllogism": {
        "premises": [
            "p ∨ q",
            "¬p",
        ],
        "premises_expr": "(p ∨ q) ∧ ¬p",
        "conclusion": "q",
        "tautology": "((p ∨ q) ∧ ¬p) → q",
    },

    "Resolution": {
        "premises": [
            "p ∨ q",
            "¬p ∨ r",
        ],
        "premises_expr": "(p ∨ q) ∧ (¬p ∨ r)",
        "conclusion": "q ∨ r",
        "tautology": "((p ∨ q) ∧ (¬p ∨ r)) → (q ∨ r)",
    },
}

selected_rule = st.selectbox(
    "Choose a rule",
    list(RULE_INFO.keys())
)

info = RULE_INFO[selected_rule]
premises_expr = info["premises_expr"]
conclusion_expr = info["conclusion"]
tautology_expr = info["tautology"]

variables = sorted(
    set(extract_variables(tautology_expr))
)

df1 = generate_truth_table_for_variables(
    premises_expr,
    variables
)

df2 = generate_truth_table_for_variables(
    conclusion_expr,
    variables
)

df3 = generate_truth_table_for_variables(
    tautology_expr,
    variables
)

display_df = pd.DataFrame()

for col in df1.columns:
    if col != "Result":
        display_df[col] = df1[col]

display_df[f"Premises: {premises_expr}"] = df1["Result"]
display_df[f"Conclusion: {conclusion_expr}"] = df2["Result"]
display_df[tautology_expr] = df3["Result"]

col1, col2 = st.columns([2, 5])

with col1:

    col3, col4 = st.columns([2,3])
    with col3:
        st.markdown("")
        st.markdown("**Premises**")
    with col4:
        st.code(
            "\n".join(info["premises"]),
            language="text"
        )

    col3, col4 = st.columns([2,3])
    with col3:
        st.markdown("")
        st.markdown("**Conclusion**")
    with col4:
        st.code(
            info["conclusion"],
            language="text"
        )

    col3, col4 = st.columns([2,3])
    with col3:
        st.markdown("")
        st.markdown("**Validity Check**")
    with col4:
        st.code(
            tautology_expr,
            language="text"
        )


with col2:

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch"
    )

st.success(f"""
{tautology_expr} is a tautology.

Therefore {selected_rule} is a valid rule of inference.
""")



