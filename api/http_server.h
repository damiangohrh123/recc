#pragma once
#include <functional>
#include <string>
#include <tuple>
#include <vector>

// Minimal single-threaded HTTP/1.1 server: enough to accept one request at a
// time, read a request line + headers + Content-Length body, dispatch it to
// a registered handler, and write back a status + body.
//
// See the top-level README's "Current Limitations" section for what's
// deliberately left out (concurrency, keep-alive, HTTPS, etc.) and why. If
// the team needs more than this later, swap this file for a real library
// such as cpp-httplib; nothing in ocr_server.cpp above the HttpServer
// interface would need to change.

struct HttpRequest {
    std::string method;  // "GET" or "POST"
    std::string path;    // e.g. "/ocr", query strings are not parsed
    std::string body;    // raw request body bytes, already fully read
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
};

class HttpServer {
public:
    using Handler = std::function<HttpResponse(const HttpRequest&)>;

    explicit HttpServer(int port);

    // Registers a handler for an exact method+path match. Checked in
    // registration order; first match wins.
    void on(const std::string& method, const std::string& path, Handler handler);

    // Binds, listens, and serves requests forever (one connection at a
    // time), on the calling thread. Does not return unless the socket
    // setup itself fails.
    void run();

private:
    int port_;
    std::vector<std::tuple<std::string, std::string, Handler>> routes_;
};
