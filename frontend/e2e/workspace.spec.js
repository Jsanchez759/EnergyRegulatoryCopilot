import { expect, test } from "@playwright/test";

import { PASSWORD, signUp } from "./accounts";
import { buildTextPdf } from "./pdf";

test("join the waitlist, get approved, sign out and sign back in", async ({ page }) => {
  const email = await signUp(page);
  await page.getByRole("button", { name: "Salir" }).click();
  await expect(page.getByRole("heading", { name: /La regulación y el mercado/ })).toBeVisible();
  await page.getByRole("link", { name: "Iniciar sesión" }).click();
  await expect(page.getByRole("heading", { name: "Qué bueno verte." })).toBeVisible();

  await page.getByLabel("Correo electrónico").fill(email);
  await page.getByLabel("Contraseña").fill(PASSWORD);
  await page.locator(".auth-submit").click();
  await expect(page.locator(".new-chat-button")).toBeVisible();
});

test("conversation sidebar can be resized and remembers its width", async ({ page }) => {
  await signUp(page);
  const pane = page.getByRole("complementary", { name: "Conversaciones" });
  const handle = page.getByRole("separator", { name: "Ajustar ancho del panel de conversaciones" });
  const before = (await pane.boundingBox()).width;

  const box = await handle.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + 200);
  await page.mouse.down();
  await page.mouse.move(box.x + 120, box.y + 200, { steps: 8 });
  await page.mouse.up();
  const widened = (await pane.boundingBox()).width;
  expect(widened).toBeGreaterThan(before + 80);

  await page.mouse.move(box.x + 2000, box.y + 200);
  await handle.focus();
  await page.keyboard.press("End");
  expect((await pane.boundingBox()).width).toBeLessThanOrEqual(481);

  await page.reload();
  await expect(page.locator(".new-chat-button")).toBeVisible();
  expect((await pane.boundingBox()).width).toBeGreaterThan(widened);

  await handle.dblclick();
  expect(Math.round((await pane.boundingBox()).width)).toBe(260);
});

test("upload a PDF and get a cited answer from it", async ({ page }) => {
  await signUp(page);

  const pdf = buildTextPdf([
    "Resolucion CREG 999 de 2030 (documento de prueba E2E).",
    "Articulo 1. El limite de autogeneracion a pequena escala del municipio ficticio de Villaprueba",
    "es de 7 megavatios.",
    "Articulo 2. El operador de red debe responder solicitudes de conexion en 9 dias habiles.",
  ]);
  await page.locator('input[type="file"]').setInputFiles({
    name: "creg-999-2030-e2e.pdf",
    mimeType: "application/pdf",
    buffer: pdf,
  });
  await page.getByRole("button", { name: "Añadir a la biblioteca" }).click();
  await expect(page.locator(".document-entry", { hasText: "creg-999-2030-e2e.pdf" })).toBeVisible({
    timeout: 120_000,
  });

  await page
    .getByLabel("Escribe tu consulta")
    .fill("Según mis documentos, ¿en cuántos días hábiles debe responder el operador de red en la Resolución CREG 999 de 2030?");
  await page.getByRole("button", { name: "Enviar consulta" }).click();

  const answer = page.locator(".message-row.assistant").last();
  await expect(page.locator(".stream-status")).toHaveText(/Completado|Error/, { timeout: 170_000 });
  await expect(answer.locator(".message-content")).not.toContainText("No se pudo completar");
  await expect(answer.locator(".message-content")).toContainText("9");

  const audit = answer.locator(".audit-panel");
  await audit.locator("summary").first().click();
  await expect(audit).not.toContainText("failed");
  await expect(audit.locator(".audit-sources")).toContainText("creg-999-2030-e2e.pdf");
  await expect(audit.locator(".audit-sources")).toContainText("Página 1");

  await page.reload();
  await expect(page.locator(".message-row.assistant").last().locator(".message-content")).toContainText("9");

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Eliminar creg-999-2030-e2e.pdf" }).click();
  await expect(page.locator(".document-entry")).toHaveCount(0);
  await expect(page.getByText("Todavía no hay documentos.")).toBeVisible();
});

test("general question is answered and persisted without document errors", async ({ page }) => {
  await signUp(page);
  await page.getByLabel("Escribe tu consulta").fill("¿Qué es la CREG? Responde en dos frases.");
  await page.getByRole("button", { name: "Enviar consulta" }).click();

  await expect(page.locator(".stream-status")).toHaveText(/Completado|Error/, { timeout: 170_000 });
  const answer = page.locator(".message-row.assistant").last();
  await expect(answer.locator(".message-content")).not.toContainText("No se pudo completar");
  await expect(answer.locator(".message-content")).not.toBeEmpty();
  await expect(answer).not.toContainText("TypeError");
  await expect(page.locator(".conversation-title").first()).toContainText("¿Qué es la CREG?");

  await page.getByLabel("Escribe tu consulta").fill("Explícame con más detalle sus funciones principales.");
  await page.getByRole("button", { name: "Enviar consulta" }).click();
  await expect(page.locator(".message-row.assistant")).toHaveCount(2);
  await expect(page.locator(".stream-status")).toHaveText(/Completado|Error/, { timeout: 170_000 });
  const followUp = page.locator(".message-row.assistant").last().locator(".message-content");
  await expect(followUp).not.toContainText("No se pudo completar");
  await expect(followUp).not.toBeEmpty();
});
