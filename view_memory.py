import sqlite3
from tabulate import tabulate  # optional, for better table view

# Connect to the database
conn = sqlite3.connect("assistant_memory.db")
c = conn.cursor()

# Fetch all saved memories
c.execute("SELECT timestamp, tags, image_path FROM memory")
rows = c.fetchall()

# Display results
print(tabulate(rows, headers=["Timestamp", "Tags", "Image Path"], tablefmt="grid"))

conn.close()
