import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from main import app
import bcrypt

client = TestClient(app)


# -------------------------
# GLOBAL MOCKS
# -------------------------
@pytest.fixture(autouse=True)
def mock_all(monkeypatch):

    # ✅ Mock DB
    class MockUsers:
        def find_one(self, query):
            if query.get("Email") == "admin@test.com":
                return {
                    "_id": "123",
                    "Email": "admin@test.com",
                    "password": bcrypt.hashpw(
                        b"1234", bcrypt.gensalt()
                    ).decode("utf-8"),
                    "Role": "Admin",
                    "Fullname": "Admin User"
                }
            if query.get("Email") == "existing@test.com":
                return {"Email": "existing@test.com"}
            return None

        def insert_one(self, data):
            class Result:
                inserted_id = "999"
            return Result()

    monkeypatch.setattr("infrastructure.database.db.Users", MockUsers())

    # ✅ Mock projects (عشان Dashboard ما يضربش)
    monkeypatch.setattr(
        "infrastructure.repositories.project_repo.get_user_projects",
        lambda user_id: []
    )


# -------------------------
# TC-A1: Valid Login
# -------------------------
def test_TC_A1_valid_login():
    response = client.post(
        "/login",
        data={"email": "admin@test.com", "password": "1234"},
        follow_redirects=False  # 🔥 مهم جدًا
    )

    assert response.status_code == 303
    assert "/Dashboard" in response.headers["location"]


# -------------------------
# TC-B1: Invalid Password
# -------------------------
def test_TC_B1_invalid_password():
    response = client.post(
        "/login",
        data={"email": "admin@test.com", "password": "wrong"},
        follow_redirects=False
    )

    assert response.status_code == 303
    assert "error=invalid" in response.headers["location"]


# -------------------------
# TC-B1: User Not Found
# -------------------------
def test_TC_B1_user_not_found():
    response = client.post(
        "/login",
        data={"email": "unknown@test.com", "password": "1234"},
        follow_redirects=False
    )

    assert response.status_code == 303
    assert "error=invalid" in response.headers["location"]


# -------------------------
# TC-A2: Signup Success
# -------------------------
def test_TC_A2_signup():
    response = client.post(
        "/signup",
        data={
            "fullname": "Test User",
            "dob": "2000-01-01",
            "nationality": "EG",
            "gender": "M",
            "email": "new@test.com",
            "password": "1234",
            "role": "User"
        },
        follow_redirects=False
    )

    assert response.status_code == 303
    assert "info=created" in response.headers["location"]


# -------------------------
# TC-B2: Signup Existing User
# -------------------------

def test_TC_B2_access_without_login():
    new_client = TestClient(app)  # 🔥 client جديد (no session)

    response = new_client.get(
        "/Dashboard",
        follow_redirects=False
    )

    assert response.status_code == 303
    assert "/Login" in response.headers["location"]