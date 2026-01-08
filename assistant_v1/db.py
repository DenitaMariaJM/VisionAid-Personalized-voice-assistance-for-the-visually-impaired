

import sqlite3

date_to_delete = "2025-12-15"

conn = sqlite3.connect("daily_summary.db")
c = conn.cursor()

c.execute("DELETE FROM summaries WHERE date=?", (date_to_delete,))

conn.commit()
conn.close()

print(f"Summary for {date_to_delete} deleted.")
