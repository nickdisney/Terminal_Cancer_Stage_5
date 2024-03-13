import sqlite3

class MemoryManager:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT,
                value TEXT
            )
        """)
        self.conn.commit()

    def store_memory(self, key, value):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO memories (key, value)
            VALUES (?, ?)
        """, (key, value))
        self.conn.commit()

    def retrieve_memory(self, key):
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM memories WHERE key = ?", (key,))
        result = cursor.fetchone()
        return result[0] if result else None

    def update_memory(self, key, value):
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE memories
            SET value = ?
            WHERE key = ?
        """, (value, key))
        self.conn.commit()

    def delete_memory(self, key):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE key = ?", (key,))
        self.conn.commit()

    def close(self):
        self.conn.close()