from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import io

from jose import jwt, JWTError
from passlib.context import CryptContext

from database import (
    users_collection, organizations_collection,
    transactions_collection, budgets_collection,
    alerts_collection, audit_logs_collection
)
from schemas import UserCreate, UserLogin, OrgCreate, Transaction, BudgetSet, AuditLog

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(title="AI FinOps Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Auth config
# ─────────────────────────────────────────────

SECRET_KEY = "finops_secret_key_2024_enterprise"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def serialize(doc):
    doc["_id"] = str(doc["_id"])
    return doc


async def write_audit(action: str, user: str, details: str = "", org_id: str = None):
    await audit_logs_collection.insert_one({
        "action": action,
        "user": user,
        "details": details,
        "org_id": org_id,
        "timestamp": datetime.utcnow()
    })


# ─────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "AI FinOps Platform v2.0 Running", "status": "healthy"}


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────

@app.post("/auth/register")
async def register(user: UserCreate):
    existing = await users_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create or find org
    org = await organizations_collection.find_one({"name": user.org_name})
    if not org:
        org_doc = {
            "name": user.org_name,
            "org_type": user.org_type,
            "created_at": datetime.utcnow()
        }
        result = await organizations_collection.insert_one(org_doc)
        org_id = str(result.inserted_id)
    else:
        org_id = str(org["_id"])

    user_doc = {
        "name": user.name,
        "email": user.email,
        "password": hash_password(user.password),
        "role": user.role,
        "org_id": org_id,
        "org_name": user.org_name,
        "org_type": user.org_type,
        "created_at": datetime.utcnow()
    }
    await users_collection.insert_one(user_doc)
    await write_audit("REGISTER", user.email, f"New user registered for org {user.org_name}", org_id)

    token = create_token({
        "sub": user.email,
        "name": user.name,
        "role": user.role,
        "org_id": org_id,
        "org_name": user.org_name,
        "org_type": user.org_type
    })
    return {"access_token": token, "token_type": "bearer", "org_type": user.org_type}


@app.post("/auth/login")
async def login(user: UserLogin):
    db_user = await users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    await write_audit("LOGIN", user.email, "User logged in", db_user.get("org_id"))

    token = create_token({
        "sub": user.email,
        "name": db_user.get("name"),
        "role": db_user.get("role"),
        "org_id": db_user.get("org_id"),
        "org_name": db_user.get("org_name"),
        "org_type": db_user.get("org_type")
    })
    return {
        "access_token": token,
        "token_type": "bearer",
        "org_type": db_user.get("org_type"),
        "role": db_user.get("role"),
        "name": db_user.get("name")
    }


@app.get("/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user


# ─────────────────────────────────────────────
# TRANSACTIONS
# ─────────────────────────────────────────────

@app.post("/transactions")
async def add_transaction(transaction: Transaction, current_user: dict = Depends(get_current_user)):
    data = transaction.dict()
    if data["date"] is None:
        data["date"] = datetime.utcnow()
    data["org_id"] = current_user.get("org_id")
    result = await transactions_collection.insert_one(data)
    await write_audit("ADD_TXN", current_user["sub"], f"Added ${transaction.amount} for {transaction.department}", current_user.get("org_id"))
    return {"message": "Transaction added", "id": str(result.inserted_id)}


@app.get("/transactions")
async def get_transactions(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    transactions = []
    async for txn in transactions_collection.find({"org_id": org_id}):
        transactions.append(serialize(txn))
    return transactions


@app.delete("/transactions/{txn_id}")
async def delete_transaction(txn_id: str, current_user: dict = Depends(get_current_user)):
    from bson import ObjectId
    await transactions_collection.delete_one({"_id": ObjectId(txn_id)})
    return {"message": "Deleted"}


# ─────────────────────────────────────────────
# CSV / EXCEL UPLOAD
# ─────────────────────────────────────────────

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    content = await file.read()
    if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))

    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    col_map = {
        "department": ["department", "dept", "division", "unit"],
        "vendor": ["vendor", "supplier", "vendor_name", "company"],
        "project": ["project", "project_name", "initiative"],
        "amount": ["amount", "cost", "spend", "total", "value", "expense"],
        "category": ["category", "type", "expense_type", "tag"],
        "description": ["description", "desc", "notes", "remarks"],
    }

    mapped = {}
    for target, options in col_map.items():
        for opt in options:
            if opt in df.columns:
                mapped[target] = opt
                break

    inserted = 0
    org_id = current_user.get("org_id")

    for _, row in df.iterrows():
        try:
            data = {
                "department": str(row.get(mapped.get("department"), "Unknown")),
                "vendor": str(row.get(mapped.get("vendor"), "N/A")),
                "project": str(row.get(mapped.get("project"), "General")),
                "amount": float(row.get(mapped.get("amount"), 0)),
                "category": str(row.get(mapped.get("category"), "General")),
                "description": str(row.get(mapped.get("description"), "")),
                "source": "csv_upload",
                "date": datetime.utcnow(),
                "org_id": org_id
            }
            await transactions_collection.insert_one(data)
            inserted += 1
        except Exception:
            continue

    await write_audit("CSV_UPLOAD", current_user["sub"], f"Uploaded {inserted} records", org_id)
    return {"message": f"Uploaded {inserted} records successfully", "records": inserted}


# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────

async def get_org_transactions(org_id: str):
    txns = []
    async for txn in transactions_collection.find({"org_id": org_id}):
        txns.append(txn)
    return txns


@app.get("/analytics/total-spend")
async def total_spend(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    txns = await get_org_transactions(org_id)
    total = sum(t.get("amount", 0) for t in txns)
    
    # Calculate month-over-month
    now = datetime.utcnow()
    this_month = sum(t.get("amount", 0) for t in txns if t.get("date") and t["date"].month == now.month and t["date"].year == now.year)
    last_month = sum(t.get("amount", 0) for t in txns if t.get("date") and t["date"].month == (now.month - 1 or 12) and t["date"].year == now.year)
    mom_change = ((this_month - last_month) / last_month * 100) if last_month > 0 else 0

    return {
        "total_spend": round(total, 2),
        "this_month": round(this_month, 2),
        "last_month": round(last_month, 2),
        "mom_change": round(mom_change, 2),
        "transaction_count": len(txns)
    }


@app.get("/analytics/spend-by-department")
async def spend_by_department(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    txns = await get_org_transactions(org_id)
    result = {}
    for t in txns:
        dept = t.get("department", "Unknown")
        result[dept] = round(result.get(dept, 0) + t.get("amount", 0), 2)
    return result


@app.get("/analytics/spend-by-vendor")
async def spend_by_vendor(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    txns = await get_org_transactions(org_id)
    result = {}
    for t in txns:
        vendor = t.get("vendor", "Unknown")
        result[vendor] = round(result.get(vendor, 0) + t.get("amount", 0), 2)
    return result


@app.get("/analytics/spend-by-project")
async def spend_by_project(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    txns = await get_org_transactions(org_id)
    result = {}
    for t in txns:
        proj = t.get("project", "General")
        result[proj] = round(result.get(proj, 0) + t.get("amount", 0), 2)
    return result


@app.get("/analytics/monthly-trend")
async def monthly_trend(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    txns = await get_org_transactions(org_id)
    monthly = {}
    for t in txns:
        date = t.get("date")
        if date:
            key = f"{date.year}-{str(date.month).zfill(2)}"
            monthly[key] = round(monthly.get(key, 0) + t.get("amount", 0), 2)
    sorted_months = sorted(monthly.items())
    return [{"month": m, "amount": a} for m, a in sorted_months]


# ─────────────────────────────────────────────
# BUDGET
# ─────────────────────────────────────────────

@app.post("/budget/set")
async def set_budget(budget: BudgetSet, current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    await budgets_collection.update_one(
        {"department": budget.department, "org_id": org_id},
        {"$set": {"department": budget.department, "amount": budget.amount, "org_id": org_id, "updated_at": datetime.utcnow()}},
        upsert=True
    )
    await write_audit("SET_BUDGET", current_user["sub"], f"Set budget for {budget.department}: ${budget.amount}", org_id)
    return {"message": "Budget set", "department": budget.department, "amount": budget.amount}


@app.get("/budget/all")
async def get_budgets(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    budgets = []
    async for b in budgets_collection.find({"org_id": org_id}):
        budgets.append(serialize(b))
    return budgets


@app.get("/budget/status")
async def budget_status(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    txns = await get_org_transactions(org_id)

    dept_spend = {}
    for t in txns:
        dept = t.get("department", "Unknown")
        dept_spend[dept] = round(dept_spend.get(dept, 0) + t.get("amount", 0), 2)

    result = []
    async for b in budgets_collection.find({"org_id": org_id}):
        dept = b["department"]
        budget_amt = b["amount"]
        spent = dept_spend.get(dept, 0)
        usage_pct = round((spent / budget_amt) * 100, 1) if budget_amt > 0 else 0
        status = "EXCEEDED" if spent > budget_amt else ("WARNING" if usage_pct > 80 else "OK")
        alert = None
        if status == "EXCEEDED":
            alert = f"{dept} exceeded budget by ${round(spent - budget_amt, 2)}. Consider reducing spend."
        elif status == "WARNING":
            alert = f"{dept} has used {usage_pct}% of budget. Monitor closely."
        result.append({
            "department": dept,
            "budget": budget_amt,
            "spent": spent,
            "remaining": round(budget_amt - spent, 2),
            "usage_pct": usage_pct,
            "status": status,
            "alert": alert
        })

    # Also add depts with spend but no budget set
    async_budgeted = [r["department"] for r in result]
    for dept, spent in dept_spend.items():
        if dept not in async_budgeted:
            result.append({
                "department": dept,
                "budget": None,
                "spent": spent,
                "remaining": None,
                "usage_pct": None,
                "status": "NO_BUDGET",
                "alert": f"{dept} has no budget set."
            })

    return result


# ─────────────────────────────────────────────
# FRAUD DETECTION (AI)
# ─────────────────────────────────────────────

@app.get("/analytics/fraud-detection")
async def fraud_detection(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    txns = await get_org_transactions(org_id)

    if len(txns) < 5:
        return {"suspicious_transactions": [], "message": "Need at least 5 transactions for analysis"}

    # Encode categorical features
    departments = [t.get("department", "Unknown") for t in txns]
    vendors = [t.get("vendor", "Unknown") for t in txns]
    amounts = [t.get("amount", 0) for t in txns]

    le_dept = LabelEncoder()
    le_vendor = LabelEncoder()

    dept_encoded = le_dept.fit_transform(departments)
    vendor_encoded = le_vendor.fit_transform(vendors)

    features = np.column_stack([amounts, dept_encoded, vendor_encoded])

    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(features)
    scores = model.score_samples(features)

    suspicious = []
    for i, pred in enumerate(predictions):
        if pred == -1:
            t = txns[i]
            t["_id"] = str(t["_id"])
            t["anomaly_score"] = round(float(scores[i]), 4)
            t["risk_level"] = "HIGH" if scores[i] < -0.3 else "MEDIUM"
            suspicious.append(t)

    return {
        "suspicious_transactions": suspicious,
        "total_analyzed": len(txns),
        "anomalies_found": len(suspicious)
    }


# ─────────────────────────────────────────────
# AI RECOMMENDATIONS
# ─────────────────────────────────────────────

@app.get("/analytics/ai-recommendations")
async def ai_recommendations(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    org_type = current_user.get("org_type", "enterprise")
    txns = await get_org_transactions(org_id)

    dept_spend = {}
    vendor_spend = {}
    vendor_count = {}

    for t in txns:
        dept = t.get("department", "Unknown")
        vendor = t.get("vendor", "Unknown")
        amt = t.get("amount", 0)
        dept_spend[dept] = dept_spend.get(dept, 0) + amt
        vendor_spend[vendor] = vendor_spend.get(vendor, 0) + amt
        vendor_count[vendor] = vendor_count.get(vendor, 0) + 1

    suggestions = []

    # Rule 1: High dept spend
    avg_dept_spend = np.mean(list(dept_spend.values())) if dept_spend else 0
    for dept, spend in dept_spend.items():
        if spend > avg_dept_spend * 1.5:
            suggestions.append({
                "category": "Overspending",
                "severity": "HIGH",
                "department": dept,
                "issue": f"{dept} is spending {round((spend/avg_dept_spend - 1)*100)}% above average",
                "suggestion": f"Review and optimize {dept} expenditures. Consider budget caps.",
                "potential_savings": round(spend - avg_dept_spend, 2)
            })

    # Rule 2: Budget check
    async for b in budgets_collection.find({"org_id": org_id}):
        dept = b["department"]
        budget = b["amount"]
        spent = dept_spend.get(dept, 0)
        if spent > budget:
            suggestions.append({
                "category": "Budget Exceeded",
                "severity": "CRITICAL",
                "department": dept,
                "issue": f"{dept} exceeded budget by ${round(spent - budget, 2)}",
                "suggestion": f"Reduce {dept} spending or request budget revision.",
                "potential_savings": round(spent - budget, 2)
            })

    # Rule 3: Top vendor dominance
    total_spend = sum(vendor_spend.values())
    for vendor, spend in vendor_spend.items():
        if total_spend > 0 and (spend / total_spend) > 0.35 and vendor not in ["N/A", "Unknown"]:
            suggestions.append({
                "category": "Vendor Concentration",
                "severity": "MEDIUM",
                "department": "All",
                "issue": f"{vendor} accounts for {round(spend/total_spend*100)}% of total vendor spend",
                "suggestion": f"Diversify vendors to reduce dependency on {vendor}. Renegotiate contract.",
                "potential_savings": round(spend * 0.15, 2)
            })

    # Rule 4: Frequent small vendor payments (possible subscription bloat)
    if org_type == "enterprise":
        for vendor, count in vendor_count.items():
            if count > 5 and vendor not in ["N/A", "Unknown"]:
                suggestions.append({
                    "category": "SaaS Optimization",
                    "severity": "LOW",
                    "department": "All",
                    "issue": f"{vendor} billed {count} times — possible duplicate SaaS subscription",
                    "suggestion": f"Audit {vendor} licenses. Consolidate or eliminate unused seats.",
                    "potential_savings": round(vendor_spend[vendor] * 0.2, 2)
                })

    # Rule 5: Government-specific policy
    if org_type == "government":
        for dept, spend in dept_spend.items():
            if spend > 50000:
                suggestions.append({
                    "category": "Policy Compliance",
                    "severity": "HIGH",
                    "department": dept,
                    "issue": f"{dept} spend of ${round(spend,2)} may require public audit",
                    "suggestion": "Ensure proper procurement documentation and public disclosure.",
                    "potential_savings": 0
                })

    return {"recommendations": suggestions, "total": len(suggestions)}


# ─────────────────────────────────────────────
# AI FORECAST
# ─────────────────────────────────────────────

@app.get("/analytics/forecast")
async def spending_forecast(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    txns = await get_org_transactions(org_id)

    if not txns:
        # Return demo forecast data
        months = []
        now = datetime.utcnow()
        for i in range(-5, 4):
            month = (now.month + i - 1) % 12 + 1
            year = now.year + ((now.month + i - 1) // 12)
            months.append({
                "month": f"{year}-{str(month).zfill(2)}",
                "actual": round(10000 + i * 1200 + np.random.normal(0, 500), 2) if i <= 0 else None,
                "forecast": round(10000 + i * 1200, 2),
                "is_forecast": i > 0
            })
        return {"forecast": months}

    # Build monthly aggregation
    monthly = {}
    for t in txns:
        date = t.get("date")
        if date:
            key = f"{date.year}-{str(date.month).zfill(2)}"
            monthly[key] = monthly.get(key, 0) + t.get("amount", 0)

    sorted_months = sorted(monthly.items())
    if len(sorted_months) < 2:
        return {"forecast": [{"month": m, "actual": a, "forecast": a, "is_forecast": False} for m, a in sorted_months]}

    # Linear regression forecast
    x = np.arange(len(sorted_months)).reshape(-1, 1)
    y = np.array([v for _, v in sorted_months])
    slope = np.polyfit(np.arange(len(y)), y, 1)
    poly = np.poly1d(slope)

    result = []
    for i, (month, actual) in enumerate(sorted_months):
        result.append({"month": month, "actual": round(actual, 2), "forecast": round(float(poly(i)), 2), "is_forecast": False})

    # Predict next 3 months
    last_month_str = sorted_months[-1][0]
    last_year, last_mo = int(last_month_str.split("-")[0]), int(last_month_str.split("-")[1])
    for j in range(1, 4):
        mo = (last_mo + j - 1) % 12 + 1
        yr = last_year + ((last_mo + j - 1) // 12)
        predicted = float(poly(len(sorted_months) - 1 + j))
        result.append({
            "month": f"{yr}-{str(mo).zfill(2)}",
            "actual": None,
            "forecast": round(max(predicted, 0), 2),
            "is_forecast": True
        })

    return {"forecast": result}


# ─────────────────────────────────────────────
# AUDIT LOGS
# ─────────────────────────────────────────────

@app.get("/audit/logs")
async def get_audit_logs(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    logs = []
    async for log in audit_logs_collection.find({"org_id": org_id}).sort("timestamp", -1).limit(50):
        logs.append(serialize(log))
    return logs


# ─────────────────────────────────────────────
# ADMIN - USERS
# ─────────────────────────────────────────────

@app.get("/admin/users")
async def get_users(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    org_id = current_user.get("org_id")
    users = []
    async for u in users_collection.find({"org_id": org_id}):
        u["_id"] = str(u["_id"])
        u.pop("password", None)
        users.append(u)
    return users


@app.put("/admin/users/{email}/role")
async def update_user_role(email: str, role: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    await users_collection.update_one({"email": email}, {"$set": {"role": role}})
    await write_audit("ROLE_CHANGE", current_user["sub"], f"Changed {email} role to {role}", current_user.get("org_id"))
    return {"message": f"Updated {email} to {role}"}


# ─────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────

@app.get("/notifications")
async def get_notifications(current_user: dict = Depends(get_current_user)):
    org_id = current_user.get("org_id")
    notifications = []

    # Budget exceeded
    budget_result = await budget_status(current_user)
    for b in budget_result:
        if b["status"] in ["EXCEEDED", "WARNING"]:
            notifications.append({
                "type": "budget",
                "severity": "CRITICAL" if b["status"] == "EXCEEDED" else "WARNING",
                "title": f"Budget {b['status']}",
                "message": b["alert"],
                "department": b["department"]
            })

    # Fraud alerts (quick check)
    txns = await get_org_transactions(org_id)
    if len(txns) >= 5:
        amounts = [[t.get("amount", 0)] for t in txns]
        model = IsolationForest(contamination=0.1, random_state=42)
        preds = model.fit_predict(amounts)
        fraud_count = sum(1 for p in preds if p == -1)
        if fraud_count > 0:
            notifications.append({
                "type": "fraud",
                "severity": "HIGH",
                "title": "Fraud Alerts Detected",
                "message": f"{fraud_count} suspicious transactions detected. Review immediately.",
                "department": "All"
            })

    return {"notifications": notifications, "count": len(notifications)}