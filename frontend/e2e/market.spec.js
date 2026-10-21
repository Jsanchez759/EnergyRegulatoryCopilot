import { expect, test } from "@playwright/test";

import { signUp } from "./accounts";

test("market page shows indicators, builds a chart and keeps the canvas", async ({ page }) => {
  await signUp(page, { name: "Prueba Mercado", prefix: "mercado" });
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
