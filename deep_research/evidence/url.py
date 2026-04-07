from urllib.parse import urlsplit, urlunsplit


def canonicalize_url(raw_url: str) -> str | None:
    """Normalize a URL for identity comparison.

    Returns None when the URL is empty, missing scheme/netloc, or has an
    invalid explicit port.
    """
    raw_url = raw_url.strip()
    if not raw_url:
        return None
    parsed = urlsplit(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    scheme = parsed.scheme.lower()
    try:
        port_value = parsed.port
    except ValueError:
        return None
    if (scheme == "http" and port_value == 80) or (
        scheme == "https" and port_value == 443
    ):
        port_value = None
    hostname = (parsed.hostname or "").lower()
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo = f"{userinfo}:{parsed.password}"
        userinfo = f"{userinfo}@"
    port = f":{port_value}" if port_value is not None else ""
    path = parsed.path or "/"
    return urlunsplit(
        (
            scheme,
            f"{userinfo}{hostname}{port}",
            path,
            parsed.query,
            parsed.fragment,
        )
    )
