import sqlite3

DB_PATH = "assistant_v2/assistant.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Show tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:")
for t in cursor.fetchall():
    print(" -", t[0])

print("\n--- interactions table ---")
cursor.execute("SELECT * FROM interactions")
rows = cursor.fetchall()

if not rows:
    print("(no rows yet)")
else:
    for row in rows:
        print(row)

conn.close()
