import streamlit as st
import pandas as pd
import itertools

from Chapter_1_utils import *

st.title("Module 3: Truth Tables and Logical Analysis")

st.markdown("""
In Module 2, we used truth tables to study conditional statements.

In this module, we build truth tables for larger expressions and use them to analyze logical propositions.
""")

st.subheader("Building Truth Tables")

st.markdown("""
A truth table shows the truth value of a proposition for every possible combination of truth values of its variables.
""")

col1, col2 = st.columns([1,2])

with col1:
    st.markdown("""
If an expression contains:

- 1 variable → 2 rows
- 2 variables → 4 rows
- 3 variables → 8 rows
- 4 variables → 16 rows
""")

with col2:

    st.markdown(" ")

    rows_df = pd.DataFrame(
        {
            "1 Variable": [2],
            "2 Variables": [4],
            "3 Variables": [8],
            "4 Variables": [16],
        },
        index=["Rows"]
    )

    st.dataframe(
        rows_df,
        width="content"
    )

st.subheader("Explore Truth Table Size")

n = st.slider(
    "Number of variables",
    min_value=1,
    max_value=4,
    value=2
)

st.success(
    f"{n} variables require {2**n} rows."
)

st.subheader("Truth Table Generator")

st.markdown("""
Enter a logical expression using the symbols below.
""")

col1, col2, col3 = st.columns([2,3,2])

with col1:
    expression = st.text_input(
        "Expression",
        value="(p ∧ q) → p"
    )
    st.caption(
        "Use only the propositional variables p, q, r, and s."
    )

    st.caption("""
    Use parentheses () to build larger expressions.
    """)

with col2:

    st.markdown("**Variables:** p, q, r, s")

    st.markdown("**Operators**")

    op1, op2, op3 = st.columns(3)

    with op1:
        st.markdown("¬&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NOT", unsafe_allow_html=True)
        st.markdown("⊕&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;XOR", unsafe_allow_html=True)

    with op2:
        st.markdown("∧&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AND", unsafe_allow_html=True)
        st.markdown("→&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IMPLIES", unsafe_allow_html=True)

    with op3:
        st.markdown("∨&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;OR", unsafe_allow_html=True)
        st.markdown("↔&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IFF", unsafe_allow_html=True)

with col3:
    st.markdown("**Examples**")

    st.code(
"""(p ∧ q) → r
¬(p ∨ q)
(p ↔ q) ∧ r""",
language="text"
    )

if st.button("Generate Truth Table"):

    df = generate_truth_table(expression)

    if df["Result"].isnull().any():
        st.error(
            "Invalid expression. Please check your expression syntax."
        )

    else:
        st.success(
            "Expression is valid."
        )

        display_df = df.rename(
            columns={
                "Result": expression
            }
        )

        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch"
        )

        classification = classify_expression(df)

        st.subheader("Classification")

        if classification == "Tautology":
            st.success(
                f"The proposition '{expression}' is a Tautology: Always True."
            )
        elif classification == "Contradiction":
            st.error(
                f"The proposition '{expression}' is a Contradiction: Always False."
            )
        else:
            st.info(
                f"The proposition '{expression}' is a Contingency: Sometimes True, Sometimes False."
            )

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### Classifying Propositions")

    st.markdown("""
A proposition may be:

- A **tautology**: always True.
- A **contradiction**: always False.
- A **contingency**: sometimes True and sometimes False.
""")

with col2:
    st.markdown("### Example Classifications")

    example_df = pd.DataFrame(
        {
            "Expression": [
                "p ∨ ¬p",
                "p ∧ ¬p",
                "p ∧ q",
            ],
            "Classification": [
                "Tautology",
                "Contradiction",
                "Contingency",
            ],
        }
    )

    st.dataframe(
        example_df,
        hide_index=True,
        width="stretch",
    )

st.info("""
Use the truth table generator above to verify the classifications of these expressions.
""")

st.subheader("Logical Equivalence")

st.markdown("""
Two propositions are logically equivalent if they have the same truth value for every 
possible assignment of truth values to their variables.
""")

st.markdown("""
To determine whether two propositions are logically equivalent:

1. Construct their truth tables.
2. Compare the final columns.
3. If the final columns are identical, the propositions are logically equivalent.
""")

col1, col2 = st.columns(2)

with col1:
    expr1 = st.text_input(
        "Expression 1",
        value="p → q",
        key="expr1"
    )
    st.caption(
    "Allowed variables: p, q, r, s    |    Allowed operators: ¬  ∧  ∨  ⊕  →  ↔")

with col2:
    expr2 = st.text_input(
        "Expression 2",
        value="¬q → ¬p",
        key="expr2"
    )
    st.caption(
    "Allowed variables: p, q, r, s    |    Allowed operators: ¬  ∧  ∨  ⊕  →  ↔")

check_equivalence = st.button(
    "Check Equivalence"
)

if check_equivalence:

    vars1 = extract_variables(expr1)
    vars2 = extract_variables(expr2)

    all_vars = sorted(
        list(set(vars1 + vars2))
    )

    invalid_expr1 = False
    invalid_expr2 = False

    # Validate Expression 1
    test_env = {
        var: True
        for var in all_vars
    }

    if eval_logic(expr1, test_env) is None:
        invalid_expr1 = True

    # Validate Expression 2
    if eval_logic(expr2, test_env) is None:
        invalid_expr2 = True

    if invalid_expr1 and invalid_expr2:
        st.error(
            "Both expressions are invalid."
        )
    elif invalid_expr1:
        st.error(
            "Expression 1 is invalid."
        )
    elif invalid_expr2:
        st.error(
            "Expression 2 is invalid."
        )
    else:

        comparison_rows = []
        equivalent = True

        for values in itertools.product(
            [True, False],
            repeat=len(all_vars)
        ):

            env = dict(
                zip(all_vars, values)
            )

            result1 = eval_logic(
                expr1,
                env
            )

            result2 = eval_logic(
                expr2,
                env
            )

            match = (result1 == result2)

            if not match:
                equivalent = False

            row = {}

            for var in all_vars:
                row[var] = (
                    "T"
                    if env[var]
                    else "F"
                )

            row[expr1] = (
                "T"
                if result1
                else "F"
            )

            row[expr2] = (
                "T"
                if result2
                else "F"
            )

            row["Match"] = (
                "✅"
                if match
                else "❌"
            )

            comparison_rows.append(row)

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        if equivalent:

            st.success(
                f"{expr1} and {expr2} are logically equivalent."
            )

        else:

            st.error(
                f"{expr1} and {expr2} are not logically equivalent."
            )

        st.dataframe(
            comparison_df,
            hide_index=True,
            width="stretch"
        )

st.info("""
Expression 1 and Expression 2 are logically equivalent if they have the same truth value for 
every assignment of truth values to their variables.

Equivalently,

```text
(Expression 1 ↔ Expression 2)
```

is a tautology. """)

st.subheader("Important Logical Equivalences")

equiv_df = pd.DataFrame(
    {
        "Expression 1": [
            "p → q",
            "¬(p ∧ q)",
            "¬(p ∨ q)",
            "p ↔ q",
        ],
        "Expression 2": [
            "¬q → ¬p",
            "¬p ∨ ¬q",
            "¬p ∧ ¬q",
            "(p → q) ∧ (q → p)",
        ],
        "Explanation": [
            "Conditional and Contrapositive",
            "DeMorgan's Law",
            "DeMorgan's Law",
            "Biconditional",
        ],
    }
)

st.dataframe(
    equiv_df,
    hide_index=True,
    width="stretch",
)

st.info("""
Use the equivalence checker above to verify these equivalences.
""")

with st.expander("Understanding the Equivalences"):

    st.markdown("""
Using

```text
p : It is raining today.
q : Alice is carrying an umbrella today.
```
""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
**Conditional and Contrapositive**

```text
p → q
```

means

```text
If it is raining today, 
then Alice is carrying an umbrella today.
```

Its contrapositive

```text
¬q → ¬p
```

means

```text
If Alice is not carrying an umbrella today,
then it is not raining today.
```
    """)

    with col2:
        st.markdown("""
**DeMorgan's Law**

```text
¬(p ∧ q) ≡ ¬p ∨ ¬q
```

means

```text
It is NOT true that

- it is raining today AND
- Alice is carrying an umbrella today.
```

This is equivalent to saying

```text
Either

- it is not raining today OR
- Alice is not carrying an umbrella today.
```
    """)
        
    with col3:
        st.markdown("""
**Biconditional**

```text
p ↔ q
```

means

```text
If it is raining today,
then Alice is carrying an umbrella today,

and

if Alice is carrying an umbrella today,
then it is raining today.
```

This is equivalent to

```text
(p → q) ∧ (q → p)
```
    """)
        
with st.expander("Programming Connection"):

    st.markdown("""
Consider

```python
if raining and not raining:
    carry_umbrella()
```

The condition

```text
raining ∧ ¬raining
```

is a contradiction. The condition is always False, 
so the code inside the if-statement will never execute.

On the other hand, consider

```python
if raining or not raining:
    carry_umbrella()
```

The condition

```text
raining ∨ ¬raining
```

is a tautology. The condition is always True, 
so the code inside the if-statement will always execute.

""")

