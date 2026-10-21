import { expect, test } from "@playwright/test";

import { signUp } from "./accounts";

test("monitor shows corpus coverage and source health", async ({ page }) => {
  await signUp(page, { name: "Prueba Corpus", prefix: "corpus" });
  await page.getByRole("link", { name: "Monitor regulatorio" }).click();

  await expect(page).toHaveURL(/\/monitor$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Qué cambió");
  await expect(page.locator(".corpus-stat").first()).not.toContainText("—");
  await expect(page.locator(".source-health-row", { hasText: "CREG" })).toContainText("normas indexadas");

  await page.getByRole("button", { name: "Incorporadas" }).click();
  await expect(page).toHaveURL(/kind=norma_incorporada/);
  await expect(page.locator(".feed-item").first()).toBeVisible();
});

test("normativa filters by status and searches inside the text", async ({ page }) => {
  await signUp(page, { name: "Prueba Corpus", prefix: "corpus" });
  await page.goto("/normativa");
  await expect(page.locator(".norm-result").first()).toBeVisible();

  await page.getByLabel("Estado", { exact: true }).selectOption("derogada");
  await expect(page).toHaveURL(/status=derogada/);
  await expect(page.locator(".norm-result .status-badge").first()).toHaveText("Derogada");

  await page.getByRole("tab", { name: "Dentro del texto" }).click();
  await page.getByLabel("Buscar dentro del texto de las normas").fill("autogeneración a pequeña escala");
  await expect(page.locator(".hit-excerpt").first()).toBeVisible({ timeout: 20_000 });
});

test("norm page shows articles and hands the norm to the assistant", async ({ page }) => {
  await signUp(page, { name: "Prueba Corpus", prefix: "corpus" });
  await page.goto("/normativa?q=174");
  await page.getByRole("link", { name: "Resolución CREG 174 de 2021" }).click();

  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Resolución CREG 174 de 2021");
  await expect(page.locator(".norm-article").first()).toBeVisible();
  await expect(page.locator(".relation-list li").first()).toContainText("Resolución CREG");

  await page.getByRole("button", { name: "Preguntar al asistente" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByLabel("Escribe tu consulta")).toHaveValue(/Resolución CREG 174 de 2021/);
});
