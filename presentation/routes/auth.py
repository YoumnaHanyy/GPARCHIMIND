from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from infrastructure.database import db

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
        # Find user in database
        user = db.Users.find_one({
            "Email": email,
            "password": password
        })

        if not user:
            # Failed login - redirect with error
            return RedirectResponse(
                url="/Login?error=invalid",
                status_code=303
            )

        # ✅ Create session with correct field names
        request.session["user"] = {
    "id": str(user["_id"]),   # 🔥 مهم
    "email": user.get("Email", ""),
    "role": user.get("Role", "User"),
    "name": user.get("Fullname", "User")
}


        # ✅ Success - redirect to dashboard
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
    email: str = Form(...),
    role: str = Form(...)
):
    try:
        # Check if user already exists
        existing_user = db.Users.find_one({"Email": email})
        
        if existing_user:
            return RedirectResponse(
                url="/Signup?error=exists",
                status_code=303
            )

        # Create new user (NOTE: In production, ALWAYS hash passwords!)
        new_user = {
            "Fullname": fullname,
            "Email": email,
            "Role": role,
            "password": "default123"  # In production, use proper password hashing
        }
        
        db.Users.insert_one(new_user)

        # Auto-login after signup
        request.session["user"] = {
            "email": email,
            "role": role,
            "name": fullname
        }

        return RedirectResponse(
            url="/Dashboard",
            status_code=303
        )

    except Exception as e:
        print(f"Signup error: {e}")
        return RedirectResponse(
            url="/Signup?error=server",
            status_code=303
        )