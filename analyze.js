exports.handler = async (event) => {
  if (event.httpMethod === "OPTIONS") {
    return corsResponse(204, "");
  }

  if (event.httpMethod !== "POST") {
    return corsResponse(405, JSON.stringify({ error: "Method not allowed" }));
  }

  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return corsResponse(
      500,
      JSON.stringify({
        error: "BACKEND_URL is not configured. Set it to your public Python backend URL.",
      })
    );
  }

  try {
    const response = await fetch(`${backendUrl.replace(/\/$/, "")}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: event.body || "{}",
    });
    const body = await response.text();
    return corsResponse(response.status, body);
  } catch (error) {
    return corsResponse(502, JSON.stringify({ error: error.message }));
  }
};

function corsResponse(statusCode, body) {
  return {
    statusCode,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Content-Type": "application/json; charset=utf-8",
    },
    body,
  };
}
