
import sqlite3

# Connect to the SQLite database.
# If student.db does not exist, SQLite creates it automatically.
# The connection context manager commits changes on success
# and rolls back the transaction if an exception occurs.
with sqlite3.connect("student.db") as connection:

    # Create a cursor to execute SQL statements.
    cursor = connection.cursor()

    # Create the STUDENT table if it does not already exist.
    # IF NOT EXISTS prevents an error when the table already exists.
    create_table_query = """
        CREATE TABLE IF NOT EXISTS STUDENT (
            NAME VARCHAR(25),
            CLASS VARCHAR(25),
            SECTION VARCHAR(25),
            MARKS INTEGER
        )
    """

    cursor.execute(create_table_query)

    # Define multiple student records as a list of tuples.
    # Each tuple represents one row in the STUDENT table.
    students = [
        ("Krish", "Data Science", "A", 90),
        ("John", "Data Science", "B", 100),
        ("Mukesh", "Data Science", "A", 86),
        ("Jacob", "DEVOPS", "A", 50),
        ("Dipesh", "DEVOPS", "A", 35),
    ]

    # Insert all records using parameterized SQL.
    # The ? placeholders safely bind Python values to SQL parameters.
    # executemany() executes the INSERT statement for each tuple.
    insert_query = """
        INSERT INTO STUDENT (NAME, CLASS, SECTION, MARKS)
        VALUES (?, ?, ?, ?)
    """

    cursor.executemany(insert_query, students)

    # Retrieve all records from the STUDENT table.
    select_query = """
        SELECT NAME, CLASS, SECTION, MARKS
        FROM STUDENT
        ORDER BY NAME
    """

    cursor.execute(select_query) ##Python sends the SQL query to SQLite. SQLite executes it and prepares the result set. At this point, you have executed the query, but you haven't explicitly retrieved the rows into a Python list.

    # Fetch and display each row returned by the SELECT query.
    print("Student records:")

    for row in cursor.fetchall(): ## cursor.fetchall() : Python retrieves all remaining rows from the query result and returns them as a list of tuples.
        print(row)

# The connection is closed automatically when the with block ends.
# Successful changes are committed; exceptions cause an automatic
# rollback of the active transaction.
