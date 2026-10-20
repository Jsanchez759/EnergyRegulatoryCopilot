import { expect, test } from "@playwright/test";

async function signIn(page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Crear cuenta" }).click();
  await page.getByLabel("Nombre").fill("Prueba Mercado");
  await page.getByLabel("Correo electrónico").fill(`mercado-${Date.now()}-${Math.random().toString(16).slice(2, 8)}@example.com`);
  await page.getByLabel("Contraseña").fill("e2e-password-segura");
  await page.locator(".auth-submit").click();
  await expect(page.getByLabel("Escribe tu consulta")).toBeVisible();
}

test("market page shows indicators, builds a chart and keeps the canvas", async ({ page }) => {
  await signIn(page);
  await page.getByRole("link", { name: "Mercado" }).click();
  await expect(page).toHaveURL(/\/mercado$/);
  await expect(page.locator(".market-tile:not(.is-loading)").first()).toBeVisible({ timeout: 60_000 });
  await expect(page.locator(".market-card .market-chart svg").first()).toBeVisible({ timeout: 90_000 });
  const before = await page.locator(".market-card").count();

  await page.getByLabel("Métrica").click();
  await page.getByLabel("Métrica").fill("volumen útil embalses");
  await page.locator(".market-options button", { hasText: "Volumen Útil diario % por Sistema" }).click();
  await page.getByRole("button", { name: "90 días" }).click();
  await page.getByRole("button", { name: "Graficar" }).click();

  await expect(page.locator(".market-card")).toHaveCount(before + 1);
  const created = page.locator(".market-card").first();
  await expect(created.locator("h3")).toContainText("Volumen Útil", { timeout: 60_000 });
  await expect(created.locator(".market-stats")).toContainText("Promedio");
  await created.getByRole("tab", { name: "Barras" }).click();
  await expect(created.locator(".market-chart rect").first()).toBeVisible();

  await page.reload();
  await expect(page.locator(".market-card")).toHaveCount(before + 1);
  await expect(page.locator(".market-card").first().getByRole("tab", { name: "Barras" })).toHaveAttribute("aria-selected", "true");
});
