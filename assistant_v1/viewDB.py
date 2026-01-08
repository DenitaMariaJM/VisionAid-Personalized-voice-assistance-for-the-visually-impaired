import sqlite3

DB_NAME = "assistant.db"

conn = sqlite3.connect(DB_NAME)
c = conn.cursor()

c.execute("SELECT * FROM interactions")
rows = c.fetchall()

if not rows:
    print("📭 The database is empty. No interactions found.")
else:
    for row in rows:
        print(row)
        print("-" * 50)

conn.close()
