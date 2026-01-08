import sqlite3

conn = sqlite3.connect("daily_summary.db")
c = conn.cursor()

c.execute("SELECT * FROM summaries")
rows = c.fetchall()

for row in rows:
    print("\n----------------------")
    print("ID:", row[0])
    print("Date:", row[1])
    print("Summary:", row[2])
    print("Key Tags:", row[3])

conn.close()
