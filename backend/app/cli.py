"""Account commands that cannot go through the web interface.

python -m app.cli admin EMAIL [--name NAME]

Promotes and activates an existing account. A new account needs a password, read from
ENERCOL_ADMIN_PASSWORD or typed at the prompt; when that variable is set, an existing
account's password is reset to it.
"""

import argparse
import asyncio
import getpass
import os

from sqlalchemy import select

from app.db import SessionFactory, engine
from app.models import User, utc_now
from app.services.auth_service import hash_password


async def make_admin(email: str, name: str | None) -> str:
    email = email.strip().casefold()
    password = os.environ.get("ENERCOL_ADMIN_PASSWORD")
    async with SessionFactory() as db:
        user = await db.scalar(select(User).where(User.email == email))
        if user is None:
            password = password or getpass.getpass("Contraseña para la nueva cuenta: ")
            if len(password) < 12:
                raise SystemExit("La contraseña debe tener al menos 12 caracteres.")
            user = User(
                email=email,
                display_name=name or email.split("@")[0],
                password_hash=hash_password(password),
            )
            db.add(user)
            outcome = "creada"
        else:
            if password:
                user.password_hash = hash_password(password)
            outcome = "actualizada"
        user.role = "admin"
        user.status = "active"
        user.approved_at = user.approved_at or utc_now()
        await db.commit()
    await engine.dispose()
    return f"Cuenta de administrador {outcome}: {email}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    admin = commands.add_parser("admin", help="Create or promote an administrator account")
    admin.add_argument("email")
    admin.add_argument("--name")
    args = parser.parse_args()
    if args.command == "admin":
        print(asyncio.run(make_admin(args.email, args.name)))


if __name__ == "__main__":
    main()
