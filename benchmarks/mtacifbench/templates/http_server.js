const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = __PORT__;
const ROOT = fs.realpathSync("__WORKSPACE__");
const ROOT_PREFIX = ROOT.endsWith(path.sep) ? ROOT : ROOT + path.sep;
const MIME_TYPES = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "application/javascript",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
};

http.createServer((req, res) => {
  let requestPath;
  try {
    requestPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  } catch (_error) {
    res.writeHead(400);
    res.end("Bad Request");
    return;
  }

  const relativePath = requestPath === "/" ? "index.html" : requestPath.replace(/^\/+/, "");
  const candidatePath = path.resolve(ROOT, relativePath);
  if (candidatePath !== ROOT && !candidatePath.startsWith(ROOT_PREFIX)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.realpath(candidatePath, (realpathError, realPath) => {
    if (realpathError || (realPath !== ROOT && !realPath.startsWith(ROOT_PREFIX))) {
      res.writeHead(realpathError ? 404 : 403);
      res.end(realpathError ? "Not Found" : "Forbidden");
      return;
    }
    fs.stat(realPath, (statError, stats) => {
      if (statError || !stats.isFile()) {
        res.writeHead(404);
        res.end("Not Found");
        return;
      }
      fs.readFile(realPath, (readError, data) => {
        if (readError) {
          res.writeHead(404);
          res.end("Not Found");
          return;
        }
        const contentType = MIME_TYPES[path.extname(realPath).toLowerCase()] || "application/octet-stream";
        res.writeHead(200, { "Content-Type": contentType });
        res.end(data);
      });
    });
  });
}).listen(PORT, "0.0.0.0", () => {
  console.log("Server running at http://localhost:" + PORT);
});
