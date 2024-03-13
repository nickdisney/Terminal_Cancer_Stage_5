import sqlite3

class KnowledgeBase:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                predicate TEXT,
                object TEXT
            )
        """)
        self.conn.commit()

    def add_fact(self, subject, predicate, object):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO facts (subject, predicate, object)
            VALUES (?, ?, ?)
        """, (subject, predicate, object))
        self.conn.commit()

    def get_facts(self, subject=None, predicate=None, object=None):
        cursor = self.conn.cursor()
        query = "SELECT * FROM facts"
        conditions = []
        if subject:
            conditions.append(f"subject = '{subject}'")
        if predicate:
            conditions.append(f"predicate = '{predicate}'")
        if object:
            conditions.append(f"object = '{object}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        cursor.execute(query)
        facts = cursor.fetchall()
        return facts

    def update_fact(self, fact_id, subject=None, predicate=None, object=None):
        cursor = self.conn.cursor()
        query = "UPDATE facts SET"
        updates = []
        if subject:
            updates.append(f"subject = '{subject}'")
        if predicate:
            updates.append(f"predicate = '{predicate}'")
        if object:
            updates.append(f"object = '{object}'")
        if updates:
            query += " " + ", ".join(updates)
            query += f" WHERE id = {fact_id}"
            cursor.execute(query)
            self.conn.commit()

    def delete_fact(self, fact_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()