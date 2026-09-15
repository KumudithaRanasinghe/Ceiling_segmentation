"""
MySQL Database Initialization & Admin Seeder
============================================
Utility script to verify your MySQL connection, create all required tables,
and optionally seed a default superadmin user account.

Usage:
    python scripts/init_mysql.py
"""
import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from app.core.config import settings
from app.core.security import hash_password
from app.domain.models.user import Base, User
from app.infrastructure.database.session import _engine, init_db, _SessionLocal
from app.domain.repositories.user_repository import UserRepository


def main():
    print(f"[*] Connecting to database using DATABASE_URL: {settings.DATABASE_URL}")

    try:
        with _engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar()
            print("[✓] Database connection successful!")
    except Exception as e:
        print(f"[!] Connection failed: {e}")
        print("\nTroubleshooting tips for MySQL Workbench / MySQL Server:")
        print("1. Ensure MySQL Server is running (`sudo systemctl status mysql`).")
        print("2. Ensure the database `ceiling_ai` exists (or create it in MySQL Workbench with `CREATE DATABASE ceiling_ai;`).")
        print("3. Check your credentials in your `.env` file:")
        print("   DATABASE_URL=\"mysql+pymysql://<user>:<password>@localhost:3306/ceiling_ai?charset=utf8mb4\"")
        sys.exit(1)

    print("[*] Creating / verifying all tables (users, segmentation_jobs, audit_logs, system_settings)...")
    init_db()
    print("[✓] Tables created successfully!")

    # Check or seed admin user
    db = _SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            print("[*] Creating default admin account...")
            admin_user = User(
                id=str(uuid.uuid4()),
                email="admin@ceiling.ai",
                username="admin",
                full_name="System Administrator",
                hashed_password=hash_password("Admin1234!"),
                roles='["admin", "user"]',
                is_active=True,
                is_superuser=True,
            )
            db.add(admin_user)
            db.commit()
            print("\n" + "=" * 50)
            print(" Default Admin Account Created:")
            print("   Username : admin")
            print("   Email    : admin@ceiling.ai")
            print("   Password : Admin1234!")
            print("   Roles    : ['admin', 'user']")
            print("=" * 50 + "\n")
        else:
            print(f"[i] Admin account '{admin_user.username}' already exists.")
    finally:
        db.close()

    print("[✓] MySQL setup complete! You can now view tables in MySQL Workbench.")


if __name__ == "__main__":
    main()
