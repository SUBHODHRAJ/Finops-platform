from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List
from enum import Enum


class OrgType(str, Enum):
    government = "government"
    enterprise = "enterprise"


class UserRole(str, Enum):
    admin = "admin"
    finance_manager = "finance_manager"
    dept_manager = "dept_manager"
    viewer = "viewer"


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    org_name: str
    org_type: OrgType
    role: UserRole = UserRole.viewer


class UserLogin(BaseModel):
    email: str
    password: str


class OrgCreate(BaseModel):
    name: str
    org_type: OrgType


class Transaction(BaseModel):
    department: str
    vendor: Optional[str] = "N/A"
    project: Optional[str] = "General"
    amount: float
    category: Optional[str] = "General"
    description: Optional[str] = ""
    source: Optional[str] = "manual"
    date: Optional[datetime] = None
    org_id: Optional[str] = None


class BudgetSet(BaseModel):
    department: str
    amount: float
    org_id: Optional[str] = None


class AuditLog(BaseModel):
    action: str
    user: str
    details: Optional[str] = ""
    org_id: Optional[str] = None
    timestamp: Optional[datetime] = None