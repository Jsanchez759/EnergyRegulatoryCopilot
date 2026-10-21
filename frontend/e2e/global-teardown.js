import { request } from "@playwright/test";

import { ADMIN, API_URL } from "./accounts";

// Every account the suite creates uses example.com; remove them so the admin list stays readable.
export default async function globalTeardown() {
  const api = await request.newContext({ baseURL: `${API_URL}/` });
  if (!(await api.post("auth/login", { data: ADMIN })).ok()) return;
  for (let round = 0; round < 20; round += 1) {
    const { users } = await (await api.get("admin/users", { params: { q: "@example.com", limit: 200 } })).json();
    const disposable = users.filter((user) => user.email !== ADMIN.email && user.email.endsWith("@example.com"));
    if (!disposable.length) break;
    for (const user of disposable) await api.delete(`admin/users/${user.id}`);
  }
  await api.dispose();
}
