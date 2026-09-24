# API Rate Limits

## Default limits
Every API key is limited per plan tier:
- Free tier: 60 requests per minute and 10,000 requests per day.
- Pro tier: 600 requests per minute and 500,000 requests per day.
- Enterprise tier: custom limits agreed in the customer contract.

## Exceeding a limit
Requests over the limit receive HTTP status 429 Too Many Requests. The response includes a `Retry-After` header giving the number of seconds to wait before retrying. Clients should use exponential backoff.

## Rate limit headers
Every response includes `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers.

## Limit increases
Pro customers can request a temporary limit increase by opening a ticket with the Platform team at least 5 business days before the expected traffic spike. Increases on the Free tier are not available.
