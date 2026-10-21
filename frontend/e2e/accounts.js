import { expect, request } from "@playwright/test";

export const PASSWORD = "e2e-password-segura";
export const API_URL = process.env.E2E_API_URL || "http://localhost:8000/api/v1";
export const ADMIN = {
  email: process.env.E2E_ADMIN_EMAIL || "e2e-admin@example.com",
  password: process.env.E2E_ADMIN_PASSWORD || "e2e-admin-password-segura",
};

let adminContext;

export async function adminApi() {
  if (!adminContext) {
    adminContext = request.newContext({ baseURL: `${API_URL}/` }).then(async (context) => {
      const response = await context.post("auth/login", { data: ADMIN });
      expect(response.ok(), "make e2e creates the admin account; run it through make").toBeTruthy();
      return context;
    });
  }
  return adminContext;
}

export async function approve(email) {
  const api = await adminApi();
  const listed = await (await api.get("admin/users", { params: { q: email } })).json();
  const user = listed.users.find((item) => item.email === email);
  expect(user, `${email} should be waiting for approval`).toBeTruthy();
  const response = await api.patch(`admin/users/${user.id}`, { data: { status: "active" } });
  expect(response.ok()).toBeTruthy();
}

export function uniqueEmail(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}@example.com`;
}

export async function signIn(page, email, password = PASSWORD) {
  await page.goto("/ingresar");
  await page.getByLabel("Correo electrónico").fill(email);
  await page.getByLabel("Contraseña").fill(password);
  await page.locator(".auth-submit").click();
  await expect(page.locator(".new-chat-button")).toBeVisible();
}

// Joins the waitlist through the page, has the admin approve it, then signs in.
export async function signUp(page, { name = "Prueba E2E", prefix = "e2e" } = {}) {
  const email = uniqueEmail(prefix);
  await page.goto("/registro");
  await page.getByLabel("Nombre").fill(name);
  await page.getByLabel("Correo electrónico").fill(email);
  await page.getByLabel("Contraseña").fill(PASSWORD);
  await page.locator(".auth-submit").click();
  await expect(page.getByRole("heading", { name: "Ya estás en la lista." })).toBeVisible();
  await approve(email);
  await signIn(page, email);
  return email;
}
