import streamlit as st
import pandas as pd

st.title("Module 1: Propositions and Logical Operations")

st.header("What Is a Proposition?")

st.markdown("""
A proposition is a declarative sentence whose value is either **True** or **False**.

The value of a proposition is called its **truth value**.
""")

col1, col2 = st.columns(2)

with col1:
    st.success("""
✅ Proposition

• 2 + 2 = 4

• 17 is even

• Today is Monday
""")

with col2:
    st.error("""
❌ Not a proposition

• What time is it?

• Close the door.

• x + 1
""")

with st.expander("Additional Note"):

    st.markdown("""
```text
2 + 1 = 10
```

is a proposition because it has a truth value. In this case, the proposition is False.

```text
x + 1 = 10
```

is not a proposition because its truth value depends on the value of x. Instead, 

```text
x + 1 = 10
```

is a predicate.

Module 5 introduces predicates and explains how a predicate becomes a proposition 
when values are assigned to its variables. """)

st.markdown("---")

st.subheader("Proposition or Not?")

examples = {
    "2 + 2 = 4": True,
    "17 is even": True,
    "What time is it?": False,
    "Close the door.": False,
    "The movie was funny.": True,
    "x + 1": False
}

col1, col2 = st.columns([3,2])

with col1:
    choice = st.selectbox(
        "Statement",
        list(examples.keys())
    )

with col2:
    user_answer = st.radio(
        "Type",
        ["Yes", "No"],
        horizontal=True
    )

if st.button("Check"):
    correct = examples[choice]

    if (user_answer == "Yes" and correct) or \
       (user_answer == "No" and not correct):
        st.success("Correct")
    else:
        st.error("Try again")

st.markdown("---")

st.subheader("Programming Connection")

st.markdown("""
Programmers often use Boolean variables.

```python
raining = True
umbrella = False
```

The programmer decides what each variable represents.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
```text
raining = True → "It is raining today."
raining = False → "It is not raining today."
```
    """)

with col2:
    st.markdown("""
```text
umbrella = True → "Alice is carrying an umbrella today."
umbrella = False → "Alice is not carrying an umbrella today."
```
    """)

st.markdown("""
A Boolean variable is itself a proposition because it always evaluates to either True or False.
""")

st.markdown("""
Boolean expressions are often used in `if` statements.

```python
if raining:
    print("Take an umbrella!")

if 9 > 10:
    print("Large value")
```

The condition after if must evaluate to either True or False. In other words, it must be a proposition. 
""")

st.markdown("""
Assume:

```python
raining = True
x = 10
name = "Alice"
```

Which of the following are propositions? 
""")

programming_examples = {
    "raining": True,
    "raining == False": True,
    "x": False,
    "x == 10": True,
    "x > 10": True,
    "name": False,
    'name == "Alice"': True,
    '"hello"': False,
}

col1, col2 = st.columns([3,2])

with col1:
    expr = st.selectbox(
        "Expression",
        list(programming_examples.keys()),
        key="programming_expr",
    )

with col2:
    answer = st.radio(
        "Proposition?",
        ["Yes", "No"],
        horizontal=True,
        key="programming_radio",
    )

if st.button("Check Programming Expression"):
    correct = programming_examples[expr]

    if (answer == "Yes" and correct) or (answer == "No" and not correct):
        st.success("Correct.")
    else:
        st.error("Try again.")

    if correct:
        st.info("This expression evaluates to either True or False.")
    else:
        st.info("This expression does not evaluate to True or False.")

st.subheader("Propositional Variables")

st.markdown("""
Logic uses the same idea. Instead of names such as
`raining` and `umbrella`, we often use shorter names.
""")

st.markdown("""
```text
p : It is raining today.
q : Alice is carrying an umbrella today.
```

These are called propositional variables.
""")

st.markdown("""
The table below shows the same propositions in programming and logic, along with their meanings.
""")

comparison_df = pd.DataFrame(
    {
        "Programming": [
            "raining = True",
            "raining = False",
            "umbrella = True",
            "umbrella = False",
        ],
        "Logic": [
            "p = True",
            "p = False",
            "q = True",
            "q = False",
        ],
        "Meaning": [
            "It is raining today.",
            "It is not raining today.",
            "Alice is carrying an umbrella today.",
            "Alice is not carrying an umbrella today.",
        ],
    }
)

st.dataframe(
    comparison_df,
    hide_index=True,
    use_container_width=True,
)

st.subheader("Logical NOT")

st.markdown("""
Suppose

```text
p : It is raining today.
```

Then

```text
¬p : It is not raining today.
```

The symbol ¬ means NOT. """)

st.info("""
Notice that `¬p` is also a proposition. Its truth value is the opposite of the truth value of `p`.

If `p` is True, then `¬p` is False. If `p` is False, then `¬p` is True.
""")

st.subheader("Logical AND")

st.markdown("""
Suppose

```text
p : It is raining today.
q : Alice is carrying an umbrella today.
```

Then

```text
p ∧ q : It is raining today AND Alice is carrying an umbrella today.
```

The symbol ∧ means AND. """)

st.info("""
Notice that `p ∧ q` is also a proposition.

Like every proposition, it has a truth value that is either True or False.
""")

st.subheader("Logical OR")

st.markdown("""
Suppose

```text
p : It is raining today.
q : Alice is carrying an umbrella today.
```

Then

```text
p ∨ q : It is raining today OR Alice is carrying an umbrella today.
```

The symbol ∨ means OR. """)

st.info("""p ∨ q is also a proposition.

In logic, OR is inclusive. If both p and q are True, then p ∨ q is also True. """)

st.subheader("Exclusive OR")

st.markdown("""
Sometimes we want exactly one proposition to be True.

```text
p ⊕ q
```

means p OR q, but not both.

This is called Exclusive OR (XOR). """)

st.info("""
`p ⊕ q` is True when exactly one of `p` and `q` is True.
""")

st.subheader("Truth Tables")

st.markdown("""
Every compound proposition has a truth value.

A truth table shows the truth value of a compound proposition for every possible combination of truth values of its variables.
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**NOT (¬)**")
    st.dataframe(
        pd.DataFrame({
            "p": ["T", "F"],
            "¬p": ["F", "T"]
        }),
        hide_index=True
    )

with col2:
    st.markdown("**AND (∧)**")
    st.dataframe(
        pd.DataFrame({
            "p": ["T", "T", "F", "F"],
            "q": ["T", "F", "T", "F"],
            "p ∧ q": ["T", "F", "F", "F"]
        }),
        hide_index=True
    )

with col3:
    st.markdown("**OR (∨)**")
    st.dataframe(
        pd.DataFrame({
            "p": ["T", "T", "F", "F"],
            "q": ["T", "F", "T", "F"],
            "p ∨ q": ["T", "T", "T", "F"]
        }),
        hide_index=True
    )

with col4:
    st.markdown("**XOR (⊕)**")
    st.dataframe(
        pd.DataFrame({
            "p": ["T", "T", "F", "F"],
            "q": ["T", "F", "T", "F"],
            "p ⊕ q": ["F", "T", "T", "F"]
        }),
        hide_index=True
    )

st.subheader("Explore Truth Values")

st.markdown("""
Suppose

```text
p : It is raining today.
q : Alice is carrying an umbrella today.
```

Change the truth values of p and q and observe the resulting compound propositions. """)

col1, col2 = st.columns(2)

with col1: p = st.checkbox( "p : It is raining today", value=True )

with col2: q = st.checkbox( "q : Alice is carrying an umbrella today", value=False )

not_p = not p
and_pq = p and q 
or_pq = p or q 
xor_pq = p != q

results_df = pd.DataFrame( 
    { "Expression": [ "¬p", "p ∧ q", "p ∨ q", "p ⊕ q", ], 
      "Meaning": [ "It is not raining today.", 
                  "It is raining today AND Alice is carrying an umbrella today.", 
                  "It is raining today OR Alice is carrying an umbrella today.", 
                  "Exactly one of 'It is raining today' or 'Alice is carrying an umbrella today' is true.", ], 
      "Truth Value": [
        "T" if not_p else "F",
        "T" if and_pq else "F",
        "T" if or_pq else "F",
        "T" if xor_pq else "F",
      ], } )

st.dataframe( results_df, hide_index=True, use_container_width=True, )

st.info("""
Observe:

- ¬p reverses the truth value of p.
- p ∧ q is True only when both p and q are True.
- p ∨ q is True when at least one of p or q is True.
- p ⊕ q is True when exactly one of p or q is True.
""")




