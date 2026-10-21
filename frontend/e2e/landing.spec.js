import { expect, test } from "@playwright/test";

import { signUp } from "./accounts";

test("landing shows live data and leads to the waitlist", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /La regulación y el mercado/ })).toBeVisible();
  await expect(page.getByRole("complementary", { name: "EnerCol en vivo" })).toContainText(/consultas regulatorias abiertas|Últimas normas publicadas/);
  await expect(page.locator(".landing-figures strong").first()).not.toHaveText("—");
  await expect(page.locator(".landing-source.is-live")).toHaveCount(3);

  await page.locator(".landing-hero").getByRole("link", { name: /Unirme a la lista de espera/ }).click();
  await expect(page).toHaveURL(/\/registro$/);
  await expect(page.getByRole("heading", { name: "Únete a la beta." })).toBeVisible();
  await expect(page.getByLabel("Perfil")).toBeVisible();
});

test("a signed-out deep link signs in and returns to the page", async ({ page }) => {
  const email = await signUp(page, { name: "Prueba Enlace", prefix: "enlace" });
  await page.getByRole("button", { name: "Salir" }).click();
  await expect(page.getByRole("heading", { name: /La regulación y el mercado/ })).toBeVisible();

  await page.goto("/normativa?status=vigente");
  await expect(page).toHaveURL(/\/ingresar$/);
  await page.getByLabel("Correo electrónico").fill(email);
  await page.getByLabel("Contraseña").fill("e2e-password-segura");
  await page.locator(".auth-submit").click();
  await expect(page).toHaveURL(/\/normativa\?status=vigente$/);
});
