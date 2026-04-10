"""Fixtures for graphsift tests."""

import pytest

from graphsift import ContextBuilder, ContextConfig, DiffSpec, OutputMode


PYTHON_SOURCE_AUTH = '''"""Auth module."""
import hashlib
import os
from typing import Optional

class AuthManager:
    """Manages user authentication."""

    def __init__(self, secret: str):
        self.secret = secret

    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify(self, password: str, hashed: str) -> bool:
        return self.hash_password(password) == hashed

def create_token(user_id: str, secret: str) -> str:
    """Create an auth token."""
    return hashlib.sha256(f"{user_id}{secret}".encode()).hexdigest()
'''

PYTHON_SOURCE_USER = '''"""User module — imports from auth."""
from auth import AuthManager, create_token

class UserService:
    """Manages users."""

    def __init__(self):
        self.auth = AuthManager(secret="supersecret")
        self._users: dict = {}

    def register(self, username: str, password: str) -> str:
        hashed = self.auth.hash_password(password)
        self._users[username] = hashed
        return create_token(username, "supersecret")

    def login(self, username: str, password: str) -> bool:
        stored = self._users.get(username)
        if not stored:
            return False
        return self.auth.verify(password, stored)
'''

PYTHON_SOURCE_TEST = '''"""Tests for auth module."""
import pytest
from auth import AuthManager, create_token

def test_hash_password():
    mgr = AuthManager(secret="s")
    hashed = mgr.hash_password("hello")
    assert len(hashed) == 64

def test_verify():
    mgr = AuthManager(secret="s")
    h = mgr.hash_password("pw")
    assert mgr.verify("pw", h)

def test_create_token():
    token = create_token("user1", "secret")
    assert len(token) == 64
'''

PYTHON_SOURCE_UTILS = '''"""Utility functions — not related to auth."""
def format_date(ts: int) -> str:
    """Format a timestamp."""
    from datetime import datetime
    return datetime.fromtimestamp(ts).isoformat()

def slugify(text: str) -> str:
    """Convert text to URL slug."""
    return text.lower().replace(" ", "-")
'''


@pytest.fixture
def source_map():
    return {
        "src/auth.py": PYTHON_SOURCE_AUTH,
        "src/user.py": PYTHON_SOURCE_USER,
        "tests/test_auth.py": PYTHON_SOURCE_TEST,
        "src/utils.py": PYTHON_SOURCE_UTILS,
    }


@pytest.fixture
def builder(source_map):
    b = ContextBuilder(ContextConfig(token_budget=50_000, output_mode=OutputMode.FULL))
    b.index_files(source_map)
    return b


@pytest.fixture
def diff_spec():
    return DiffSpec(
        changed_files=["src/auth.py"],
        query="Review the authentication changes",
        commit_message="refactor: improve auth token generation",
    )
