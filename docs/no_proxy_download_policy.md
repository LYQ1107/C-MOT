# Non-proxy download policy

Large data, annotation, weight, and dependency downloads are allowed only through the isolated child process in `cmot/download.py`.

The child environment is constructed from an allow-list rather than inherited.  It records only the presence of parent proxy variable names, never their values.  `curl` is invoked with:

```text
-q --proxy "" --noproxy "*" --fail --show-error --location
--connect-timeout 15 --max-redirs 5 --retry 2
--proto =https --proto-redir =https
```

Before any download, the module audits DNS, interfaces, default routes, policy rules, tunnel-like links, and proxychains processes.  If the direct route is not verified, the result is `BLOCKED_NO_VERIFIED_DIRECT_ROUTE` and the file is not attempted.  A failed direct transfer is `DIRECT_DOWNLOAD_FAILED_NO_PROXY_FALLBACK`; it is never retried through a proxy or mirror.

The task's official BDD label URL audit was `NOT_RUN` for download because the purified child could not resolve the host without the parent proxy path.  Existing local BDD MOT assets were used instead.

