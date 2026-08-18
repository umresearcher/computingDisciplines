import streamlit as st
import pandas as pd

st.title("Module 2: Conditional Statements and Logical Equivalence")

st.markdown("""
In Module 1, we learned how to represent and combine propositions.

In this module, we study conditional statements,
their related forms, and logical equivalence.
""")

st.subheader("Conditional Statements")

st.markdown("Suppose")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
```text
p : It is raining today.
```
""")

with col2: st.markdown("""
```text
q : Alice is carrying an umbrella today.
```
""")

st.markdown(""" Then
```text
p → q
```

means

```text
If it is raining today, then Alice is carrying an umbrella today.
```

The symbol → means implies. """)

st.subheader("Related Conditional Statements") 

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
```text
p : It is raining today.
```
""")

with col2: st.markdown("""
```text
q : Alice is carrying an umbrella today.
```
""")

related_df = pd.DataFrame( 
    { "Name": [ "Conditional", "Converse", "Inverse", "Contrapositive", ], 
     "Form": [ "p → q", "q → p", "¬p → ¬q", "¬q → ¬p", ], 
     "Meaning": [ "If it is raining today, then Alice is carrying an umbrella today.", 
                 "If Alice is carrying an umbrella today, then it is raining today.", 
                 "If it is not raining today, then Alice is not carrying an umbrella today.", 
                 "If Alice is not carrying an umbrella today, then it is not raining today.", ],
     } ) 

st.dataframe( related_df, hide_index=True, use_container_width=True, )

st.info("""
Truth tables help us determine how these four statements are related.
""")

st.subheader("Explore Conditional Statements")

col1, col2 = st.columns([1,2])

with col1:

    #add tiny amount of vertical space on the left column
    st.markdown("""    """)

    p = st.checkbox(
        "p : It is raining today.",
        value=True,
    )

    q = st.checkbox(
        "q : Alice is carrying an umbrella today.",
        value=True,
    )

with col2:

    conditional = (not p) or q
    converse = (not q) or p
    inverse = p or (not q)
    contrapositive = q or (not p)

    results_df = pd.DataFrame(
        {
            "Statement": [
                "Conditional",
                "Converse",
                "Inverse",
                "Contrapositive",
            ],
            "Form": [
                "p → q",
                "q → p",
                "¬p → ¬q",
                "¬q → ¬p",
            ],
            "Truth Value": [
                "T" if conditional else "F",
                "T" if converse else "F",
                "T" if inverse else "F",
                "T" if contrapositive else "F",
            ],
        }
    )

    st.dataframe(
        results_df,
        hide_index=True,
        width="stretch",
    )

st.subheader("Truth Tables of Related Conditional Statements")

truth_df = pd.DataFrame(
    {
        "p": ["T", "T", "F", "F"],
        "q": ["T", "F", "T", "F"],
        "Conditional\np → q": ["T", "F", "T", "T"],
        "Converse\nq → p": ["T", "T", "F", "T"],
        "Inverse\n¬p → ¬q": ["T", "T", "F", "T"],
        "Contrapositive\n¬q → ¬p": ["T", "F", "T", "T"],
    }
)

st.dataframe(
    truth_df,
    hide_index=True,
    width="stretch",
)

st.info("""
Compare the columns carefully.

Which two columns are identical?
""")

st.success("""
Conditional and Contrapositive have identical truth tables.

Converse and Inverse have identical truth tables.
""")

st.subheader("Logical Equivalence")

st.markdown("""
Two propositions are logically equivalent if they have the same truth table.
""")

st.success("""
p → q  ≡  ¬q → ¬p

q → p  ≡  ¬p → ¬q
""")

with st.expander("Why do contrapositives matter?"):

    st.markdown("""
Consider the statement

```text
If n² is even, then n is even.
```

Its contrapositive is

```text
If n is odd, then n² is odd.
```

Since a conditional statement and its contrapositive are logically equivalent, proving the contrapositive also proves the original statement.

To prove the contrapositive, suppose n is odd. In other words, 

```text
n = 2k + 1 for some integer k.
```

Therefore

```text
n² = (2k + 1)²
   = 4k² + 4k + 1
   = 2(2k² + 2k) + 1
```

Since n² = 2(2k² + 2k) + 1 has the form 2m + 1, n² is odd.

Therefore:

```text
If n is odd, then n² is odd.
```

Since this is the contrapositive of

```text
If n² is even, then n is even,
```

and a conditional statement is logically equivalent to its contrapositive, 
the original statement is also true.
""")

st.subheader("Biconditional Statements")

st.markdown("Suppose")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
```text
p : It is raining today.
```
""")

with col2: 
    st.markdown("""
```text
q : Alice is carrying an umbrella today.
```
""")

st.markdown("""
Then

```text
p ↔ q
```

means

```text
p if and only if q
```

The symbol ↔ means if and only if (IFF). """)

st.info(""" A biconditional is true when p and q have the same truth value. """)

biconditional_df = pd.DataFrame(
    {
        "p": ["T", "T", "F", "F"],
        "q": ["T", "F", "T", "F"],
        "p ↔ q": ["T", "F", "F", "T"],
    }
)

st.dataframe(
    biconditional_df,
    hide_index=True,
    width="content",
)

st.success("""
p ↔ q  ≡  (p → q) ∧ (q → p)
""")

st.info("""
In the next module, we will build truth tables for larger expressions and determine whether they are tautologies, contradictions, or contingencies.
""")


