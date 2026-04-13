from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from infrastructure.database import db
import bcrypt
from datetime import datetime

router = APIRouter()
templates = Jinja2Templates(directory="presentation/templates")

# -------------------------
# LOGIN POST
# -------------------------
@router.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    try:

        user = db.Users.find_one({"Email": email})

        if not user:
            return RedirectResponse(
                url="/Login?error=invalid",
                status_code=303
            )

        if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
            return RedirectResponse(
                url="/Login?error=invalid",
                status_code=303
            )

        request.session["user"] = {
            "id": str(user["_id"]),
            "email": user.get("Email", ""),
            "role": user.get("Role", "User"),
            "name": user.get("Fullname", "User")
        }

        return RedirectResponse(
            url="/Dashboard",
            status_code=303
        )

    except Exception as e:
        print(f"Login error: {e}")
        return RedirectResponse(
            url="/Login?error=server",
            status_code=303
        )

# -------------------------
# SIGNUP POST
# -------------------------
@router.post("/signup")
async def signup(
    request: Request,
    fullname: str = Form(...),
    dob: str = Form(...),
    nationality: str = Form(...),
    gender: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    custom_role: str = Form(None)
):
    try:

        # check if exists
        existing_user = db.Users.find_one({"Email": email})

        if existing_user:
            return RedirectResponse(
                url="/Signup?error=exists",
                status_code=303
            )

        # hash password
        hashed_password = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        )

        now = datetime.utcnow()
        if role == "others" and custom_role:
           role = custom_role

        new_user = {
    "Fullname": fullname,
    "Date_of_birth": dob,
    "Nationality": nationality,
    "Gender": gender,
    "Email": email,
    "password": hashed_password.decode("utf-8"),
    "Role": role,
    "created_at": now,
    "updated_at": now
}

        result = db.Users.insert_one(new_user)

        # create session
        request.session["user"] = {
            "id": str(result.inserted_id),
            "email": email,
            "role": role,
            "name": fullname
        }

        return RedirectResponse(
            url="/Login?info=created",
            status_code=303
        )

    except Exception as e:
        print("Signup error:", e)
        return RedirectResponse(
            url="/Signup?error=server",
            status_code=303
        )