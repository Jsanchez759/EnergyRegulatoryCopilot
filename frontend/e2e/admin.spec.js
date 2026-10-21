import { expect, request, test } from "@playwright/test";

import { ADMIN, API_URL, PASSWORD, signIn, signUp, uniqueEmail } from "./accounts";

test("admin approves a waitlisted account from the panel", async ({ page }) => {
  const email = uniqueEmail("espera");
  const visitor = await request.newContext({ baseURL: `${API_URL}/` });
  const registered = await visitor.post("auth/register", {
    data: { display_name: "Persona en espera", email, password: PASSWORD, company: "Comercializadora E2E", profile: "comercializador" },
  });
  expect((await registered.json()).status).toBe("waitlist");
  expect((await visitor.post("auth/login", { data: { email, password: PASSWORD } })).status()).toBe(403);

  await signIn(page, ADMIN.email, ADMIN.password);
  await page.getByRole("link", { name: "Administración" }).click();
  await expect(page.getByRole("heading", { name: /Cómo va/ })).toBeVisible();
  await expect(page.locator(".admin-stats .corpus-stat")).toHaveCount(4);

  await page.locator(".admin-nav").getByRole("link", { name: "Usuarios" }).click();
  await page.getByRole("button", { name: /En espera/ }).click();
  await page.getByLabel("Buscar usuarios").fill(email);
  const row = page.locator(".admin-table tbody tr", { hasText: email });
  await expect(row).toContainText("Comercializadora E2E");
  await expect(row).toContainText("En espera");
  await row.getByRole("button", { name: "Aprobar" }).click();
  await expect(row).toHaveCount(0);

  await page.getByRole("button", { name: /^Todos/ }).click();
  await page.locator(".admin-table tbody tr", { hasText: email }).getByRole("button", { name: "Persona en espera" }).click();
  const drawer = page.getByRole("complementary", { name: "Detalle del usuario" });
  await expect(drawer).toContainText("Activo");
  await expect(drawer).toContainText("Comercializador");
  await expect(drawer.getByRole("button", { name: "Suspender" })).toBeVisible();

  expect((await visitor.post("auth/login", { data: { email, password: PASSWORD } })).ok()).toBeTruthy();
  await visitor.dispose();

  await page.locator(".admin-nav").getByRole("link", { name: "Actividad" }).click();
  await expect(page.locator(".admin-audit")).toContainText(`aprobó a ${email}`);
});

test("members do not see the admin section", async ({ page }) => {
  await signUp(page, { name: "Prueba Miembro", prefix: "miembro" });
  await expect(page.getByRole("link", { name: "Investigación" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Administración" })).toHaveCount(0);
  await page.goto("/admin");
  await expect(page).toHaveURL(/\/$/);
});
