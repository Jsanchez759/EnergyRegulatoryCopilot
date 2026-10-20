from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    display_name: str = Field(min_length=2, max_length=100)
    email: EmailStr = Field(max_length=254)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("display_name")
    @classmethod
    def strip_display_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("El nombre debe tener al menos dos caracteres.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: str
    display_name: str
    email: EmailStr
    created_at: str


class ProfileUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=2, max_length=100)
    email: EmailStr | None = Field(default=None, max_length=254)
    current_password: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("display_name")
    @classmethod
    def strip_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if len(value) < 2:
            raise ValueError("El nombre debe tener al menos dos caracteres.")
        return value


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)
