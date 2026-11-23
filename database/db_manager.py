"""
================================================================================
PestTrack AI - SQLite Database Manager
================================================================================
"""

import sqlite3
import os
import shutil
import datetime
from typing import List, Dict, Any, Optional
import config

class DatabaseManager:
    """Manages SQLite persistent storage for PestTrack AI analysis history."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes database tables if they do not exist."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Analysis History Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,       -- 'detection', 'classification', 'segmentation', 'full'
                    image_name TEXT,
                    original_image_path TEXT,
                    result_image_path TEXT,
                    pest_count INTEGER DEFAULT 0,
                    primary_pest_species TEXT,
                    confidence REAL DEFAULT 0.0,
                    damage_percentage REAL DEFAULT 0.0,
                    severity_level TEXT DEFAULT 'Low', -- 'Low', 'Medium', 'High'
                    notes TEXT
                );
            """)

            # Pest Detections Table (bounding boxes)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pest_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    pest_class TEXT,
                    confidence REAL,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    FOREIGN KEY(analysis_id) REFERENCES analysis_history(id) ON DELETE CASCADE
                );
            """)

            # System settings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            conn.commit()

        # Automatically clean up any existing duplicate entries
        self.deduplicate_records()

    def deduplicate_records(self):
        """Cleans up duplicate records by keeping only the latest entry per image_name."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    DELETE FROM pest_detections 
                    WHERE analysis_id NOT IN (
                        SELECT MAX(id) FROM analysis_history GROUP BY image_name
                    );
                """)
                cur.execute("""
                    DELETE FROM analysis_history 
                    WHERE id NOT IN (
                        SELECT MAX(id) FROM analysis_history GROUP BY image_name
                    );
                """)
                conn.commit()
        except Exception as e:
            print(f"Error deduplicating database records: {e}")

    def insert_analysis(self,
                        analysis_type: str,
                        image_name: str,
                        original_image_path: str,
                        result_image_path: str = "",
                        pest_count: int = 0,
                        primary_pest_species: str = "None",
                        confidence: float = 0.0,
                        damage_percentage: float = 0.0,
                        severity_level: str = "Low",
                        notes: str = "",
                        detections: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Inserts or updates an analysis record avoiding duplicates.
        If an image with the same image_name was already analyzed, updates the existing row.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Prevent duplicates: Check if image_name already exists in database
            cur.execute("SELECT id FROM analysis_history WHERE image_name = ?", (image_name,))
            existing = cur.fetchone()

            if existing:
                analysis_id = existing["id"]
                cur.execute("""
                    UPDATE analysis_history SET
                        timestamp = ?,
                        analysis_type = ?,
                        original_image_path = ?,
                        result_image_path = ?,
                        pest_count = ?,
                        primary_pest_species = ?,
                        confidence = ?,
                        damage_percentage = ?,
                        severity_level = ?,
                        notes = ?
                    WHERE id = ?
                """, (
                    timestamp, analysis_type, original_image_path,
                    result_image_path, pest_count, primary_pest_species,
                    round(confidence, 2), round(damage_percentage, 2), severity_level, notes,
                    analysis_id
                ))
                # Delete old detections for this analysis_id before re-inserting
                cur.execute("DELETE FROM pest_detections WHERE analysis_id = ?", (analysis_id,))
            else:
                cur.execute("""
                    INSERT INTO analysis_history (
                        timestamp, analysis_type, image_name, original_image_path,
                        result_image_path, pest_count, primary_pest_species,
                        confidence, damage_percentage, severity_level, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, analysis_type, image_name, original_image_path,
                    result_image_path, pest_count, primary_pest_species,
                    round(confidence, 2), round(damage_percentage, 2), severity_level, notes
                ))
                analysis_id = cur.lastrowid

            if detections and analysis_id:
                for d in detections:
                    cur.execute("""
                        INSERT INTO pest_detections (
                            analysis_id, pest_class, confidence,
                            bbox_x1, bbox_y1, bbox_x2, bbox_y2
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        analysis_id,
                        d.get("class", d.get("class_name", "Pest")),
                        round(d.get("confidence", 0.0), 2),
                        d.get("bbox", [0, 0, 0, 0])[0],
                        d.get("bbox", [0, 0, 0, 0])[1],
                        d.get("bbox", [0, 0, 0, 0])[2],
                        d.get("bbox", [0, 0, 0, 0])[3]
                    ))
            conn.commit()
            return analysis_id

    def get_all_analyses(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Retrieves history records sorted by latest first."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single record by its ID."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def delete_analysis(self, analysis_id: int) -> bool:
        """Deletes a record and its detections."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM pest_detections WHERE analysis_id = ?", (analysis_id,))
            cur.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Calculates dashboard summary metrics."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS total_count FROM analysis_history")
            total_analyzed = cur.fetchone()["total_count"]

            cur.execute("SELECT SUM(pest_count) AS total_pests FROM analysis_history")
            res_pests = cur.fetchone()["total_pests"]
            total_pests = res_pests if res_pests is not None else 0

            cur.execute("SELECT COUNT(*) AS high_count FROM analysis_history WHERE severity_level = 'High'")
            high_severity_cases = cur.fetchone()["high_count"]

            cur.execute("SELECT * FROM analysis_history ORDER BY id DESC LIMIT 5")
            recent_rows = [dict(r) for r in cur.fetchall()]

            # Infestation breakdown for charts
            cur.execute("SELECT severity_level, COUNT(*) AS count FROM analysis_history GROUP BY severity_level")
            severity_counts = {r["severity_level"]: r["count"] for r in cur.fetchall()}

            return {
                "total_analyzed": total_analyzed,
                "total_pests": total_pests,
                "high_severity_cases": high_severity_cases,
                "recent_analyses": recent_rows,
                "severity_counts": severity_counts
            }

    def backup_database(self, destination_path: str) -> bool:
        """Creates a backup copy of the SQLite database."""
        try:
            shutil.copy2(self.db_path, destination_path)
            return True
        except Exception as e:
            print(f"Database backup error: {e}")
            return False

    def restore_database(self, source_path: str) -> bool:
        """Restores database from a backup file."""
        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, self.db_path)
                return True
            return False
        except Exception as e:
            print(f"Database restore error: {e}")
            return False
