import streamlit as st
import pandas as pd
from Chapter_6_utils import *

st.title(
    "Module 7: N-ary Relations ↔ N-column Tables"
)

st.markdown("""
In the previous modules, we studied binary
relations.

A binary relation R defined over sets A and B
is a subset of A × B.

Its elements (or tuples) have the form:

(a, b)

where a ∈ A and b ∈ B.

In this module, we explore **n-ary relations**,
whose elements (or tuples) have the form:

(a₁, a₂, ..., aₙ)

N-ary relations provide the mathematical
foundation for relational databases, SQL,
and Datalog.
""")

st.header(
    "N-ary Relations"
)

st.subheader(
    "Step 1: A Ternary Relation"
)

enrollment_df = pd.DataFrame(
    [
        ["Alice", "MTH 230", "A"],
        ["Bob", "CSC 379", "B"],
        ["Charlie", "MTH 230", "A"],
        ["Alice", "MTH 328", "A-"]
    ],
    columns=[
        "Student",
        "Course",
        "Grade"
    ]
)

st.dataframe(
    enrollment_df,
    hide_index=True,
    use_container_width=True
)

st.info("""
This table represents a ternary relation:

Enrollment(Student, Course, Grade)

defined over:

• Students = {Alice, Bob, Charlie, ...},
  where Student ∈ Students

• Courses = {MTH 230, MTH 328,
  CSC 379, ...},
  where Course ∈ Courses

• Grades = {A, A-, B, ...},
  where Grade ∈ Grades

Each row is a tuple of the form:

(Student, Course, Grade)
""")

st.subheader(
    "Step 2: Relations as Tables"
)

st.markdown("""
An n-ary relation can be represented as a table with n columns.

- Columns represent attributes.
- Rows represent tuples.
- The order of rows does not matter.
- Duplicate tuples are not allowed.
""")

st.info("""
The ternary relation

Enrollment(Student, Course, Grade)

can be represented as the 3-column table above.
""")

st.subheader(
    "Step 3: Selection (Choosing Rows)"
)

st.markdown("""
Selection chooses rows that satisfy a condition.

For example, we may wish to select all tuples
for which:

Grade = A
""")

selected_grade = st.selectbox(
    "Select Grade",
    sorted(
        enrollment_df["Grade"].unique()
    )
)

selected_df = enrollment_df[
    enrollment_df["Grade"] == selected_grade
]

st.markdown(
    "### Result of Selection"
)

st.dataframe(
    selected_df,
    hide_index=True,
    use_container_width=True
)

st.info(
    f"""
SQL

SELECT *
FROM Enrollment
WHERE Grade = '{selected_grade}';
"""
)

st.info(
    f"""
Datalog

Result(s, c, g) :-
    Enrollment(s, c, g),
    g = '{selected_grade}'.
"""
)

st.caption("""
Interpretation of the Datalog Rule:

Find all values of s, c, and g such that

• Enrollment(s, c, g) is true, and
• g = '{selected_grade}'.

Each such tuple belongs to Result.
""")

st.success("""
Selection chooses rows that satisfy
a condition.
""")

st.subheader(
    "Step 4: Projection (Choosing Columns)"
)

st.markdown("""
Projection chooses columns from a relation.
""")

selected_columns = st.multiselect(
    "Select Attributes",
    enrollment_df.columns.tolist(),
    default=["Student"]
)

if selected_columns:
    projection_df = enrollment_df[
        selected_columns
    ].drop_duplicates()

    st.markdown(
        "### Result of Projection"
    )

    st.dataframe(
        projection_df,
        hide_index=True,
        use_container_width=True
    )

    columns_sql = ", ".join(
        selected_columns
    )

    st.info(
        f"""
    SQL

    SELECT {columns_sql}
    FROM Enrollment;
    """
    )

    var_map = {
        "Student": "s",
        "Course": "c",
        "Grade": "g"
    }

    selected_vars = [
        var_map[col]
        for col in selected_columns
    ]

    result_vars = ", ".join(
        selected_vars
    )

    datalog_rule = (
        f"Result({result_vars}) :- "
        f"Enrollment(s, c, g)."
    )

    st.info(
        f"""
    Datalog

    {datalog_rule}
    """
    )

    st.caption(
        f"""
    Interpretation of the Datalog Rule:

    Find all values of

    {", ".join(selected_vars)}

    such that

    Enrollment(s, c, g) is true.

    Each resulting tuple belongs to Result.
    """
    )

    st.info("""
    Notice that duplicate tuples have been removed.

    Relations are sets, so duplicate tuples are
    not allowed.
    """)

st.success("""
Projection chooses columns from a relation.
""")

st.subheader(
    "Step 5: Join (Combining Relations)"
)

st.markdown("""
A join combines tuples from two relations
using a shared attribute.
""")

students_df = pd.DataFrame(
    [
        [1, "Alice"],
        [2, "Bob"],
        [3, "Charlie"]
    ],
    columns=[
        "StudentID",
        "Student"
    ]
)

enrollments_df = pd.DataFrame(
    [
        [1, "MTH 230"],
        [2, "CSC 379"],
        [3, "MTH 230"],
        [1, "MTH 328"]
    ],
    columns=[
        "StudentID",
        "Course"
    ]
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        "### Students"
    )

    st.dataframe(
        students_df.astype(str),
        hide_index=True,
        use_container_width=True
    )

with col2:

    st.markdown(
        "### Enrollments"
    )

    st.dataframe(
        enrollments_df.astype(str),
        hide_index=True,
        use_container_width=True
    )

joined_df = students_df.merge(
    enrollments_df,
    on="StudentID"
)

display_df = joined_df[
    ["Student", "Course"]
]

st.caption(
    "The join matches tuples having the same StudentID."
)

st.markdown(
    "### Result of Join"
)

st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True
)

st.info("""
SQL

SELECT Student, Course
FROM Students
JOIN Enrollments
ON Students.StudentID =
   Enrollments.StudentID;
""")

st.info("""
Datalog

Result(s, c) :-
    Students(id, s),
    Enrollments(id, c).
""")

st.caption("""
Interpretation of the Datalog Rule:

Find all values of id, s, and c such that

• Students(id, s) is true, and
• Enrollments(id, c) is true.

Each resulting tuple (s, c)
belongs to Result.
""")

st.success("""
A join combines tuples from two relations
using a shared attribute.
""")

st.subheader(
    "Step 6: Datalog and Reachability"
)

st.markdown("""
Datalog can express relationships that
involve paths and reachability.

Let us examine **reachability**
in a directed graph.
""")

st.markdown("""
Consider the relation

**Edge(Source, Destination)**.
""")

edge_df = pd.DataFrame(
    [
        [1, 2],
        [2, 3],
        [3, 4]
    ],
    columns=[
        "Source",
        "Destination"
    ]
)

st.dataframe(
    edge_df.astype(str),
    hide_index=True,
    use_container_width=True
)

st.info("""
Datalog

Reachable(s, d) :-
    Edge(s, d).

Reachable(s, d) :-
    Edge(s, m),
    Reachable(m, d).
""")

st.caption("""
Interpretation of the Datalog Rules:

Rule 1: If there is a direct edge from s to d,
then d is reachable from s.

Rule 2: If there is an edge from s to m and d is
reachable from m, then d is also reachable
from s.
""")

st.info("""
These rules compute reachability by repeatedly
finding longer and longer paths.

This is the same idea we studied in:

- Relation Powers (R², R³, ...)
- Matrix Powers (A², A³, ...)
- Transitive Closure (R⁺)
""")

reachable_df = pd.DataFrame(
    [
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 3],
        [2, 4],
        [1, 4]
    ],
    columns=[
        "Source",
        "Destination"
    ]
)

st.markdown(
    "### Reachable Relation"
)

st.dataframe(
    reachable_df,
    hide_index=True,
    use_container_width=True
)

st.success("""
Datalog can express recursive relationships
such as reachability and transitive closure.
""")

with st.expander(
    "Additional Material: Relation Powers in Datalog"
):
    st.markdown("""
    Recall from Module 4:

    - R² contains all pairs connected by a path of length exactly 2.
    - R³ contains all pairs connected by a path of length exactly 3.
    - R⁺ is the transitive closure.

    These relations can also be expressed using
    Datalog.
    """)
    col1, col2 = st.columns(2)

    with col1:

        st.info("""
    R²

    R2(s, d) :-
        Edge(s, m),
        Edge(m, d).
    """)

    with col2:

        st.caption("""
    Interpretation

    R2 computes R², that is,
    all pairs (s, d) connected
    by a path of length 2.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
    RLe2

    RLe2(s, d) :-
        Edge(s, d).

    RLe2(s, d) :-
        R2(s, d).
    """)

    with col2:

        st.caption("""
    Interpretation

    RLe2 computes all pairs
    (s, d) connected by a path
    of length at most 2.
    """)

    st.markdown("""
    The following Datalog rules all compute **R³**,
    that is, all pairs (s, d) connected by a path
    of length exactly 3.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.code("""
    R3(s, d) :-
        Edge(s, m1),
        Edge(m1, m2),
        Edge(m2, d).
    """)

    with col2:

        st.code("""
    R3(s, d) :-
        Edge(s, m),
        R2(m, d).
    """)

    with col3:

        st.code("""
    R3(s, d) :-
        R2(s, m),
        Edge(m, d).
    """)

    st.info("""
    The recursive definition of Reachable computes paths of all
    lengths and therefore captures the transitive
    closure.
    """)

