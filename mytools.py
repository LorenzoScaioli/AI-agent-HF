from typing import List
from urllib.parse import parse_qs, unquote, urlparse
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    WebBaseLoader,
    WikipediaLoader,
)
import os
from typing import Union
import wolframalpha

from langchain_core.tools import tool


# Tool to perform basic arithmetic operations
@tool
def calculator(a: Union[int, float], b: Union[int, float], operation: str) -> float:
    """
    Perform basic arithmetic between two numeric values.

    Supported operations:
      - add (alias: sum)
      - subtract
      - multiply
      - divide (alias: div)
      - modulus (alias: mod)

    Args:
        a: First operand (int or float).
        b: Second operand (int or float).
        operation: Operation name as string.

    Returns:
        The result as float.

    Error Handling (Raises ValueError):
        Returns a structured error message if an unsupported
        operation is provided or if division by zero is attempted.
    """
    if operation == "add" or operation == "sum":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide" or operation == "div":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        result = a / b
    elif operation == "modulus" or operation == "mod":
        result = a % b
    else:
        err_msg = f"Unsupported operation: {operation}. "
        err_msg += "Use one of: add, subtract, multiply, divide, modulus "
        err_msg += "or try the 'wolfram_query' tool for complex math."
        raise ValueError(err_msg)

    return float(result)


# Tool to compute complex math queries using Wolfram Alpha API
@tool
def wolfram_query(expression: str) -> str:
    """
    Compute complex mathematical expressions or queries via the
    Wolfram Alpha API.

    Args:
        expression: A natural-language or Wolfram-style math query,
                    e.g. "integrate x^2", "solve x^2 + 3x + 2 = 0",
                    "derivative of sin(x)".

    Returns:
        The API’s primary result as text, or a structured error message.

    Error Handling:
        - Missing API key: returns an error string.
        - No result found: informs the user.
        - API errors or exceptions: returns the exception message.
    """
    app_id = os.getenv("WOLFRAM_APP_ID")
    if not app_id:
        return (
            "[Tool: wolfram_query ERROR] Missing WOLFRAM_APP_ID environment variable."
        )

    try:
        client = wolframalpha.Client(app_id)
        res = client.query(expression)
        # Attempt to get the first pod’s plaintext result
        pod = next(res.results, None)
        if pod is None or not hasattr(pod, "text") or not pod.text:
            return f"[Tool: wolfram_query] No result for query: '{expression}'."
        return f"[Tool: wolfram_query] {pod.text}"
    except StopIteration:
        return f"[Tool: wolfram_query] No results returned for '{expression}'."
    except Exception as e:
        return f"[Tool: wolfram_query ERROR] {str(e)}"


# Tool to search Wikipedia
@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia (max 2 results)."""
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return "\n\n".join(
        [f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
    )


# to searhc on the web using DuckDuckGo
@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Return search content and domain.

    Args:
        query: The search query."""
    try:
        domain = None
        web_search = WebSearchTool()

        if "site:" in query:
            query, domain = query.split("site:", 1)

        search_result = web_search.search(query.strip())
        if domain:
            search_result += f"\nDomain filter: {domain.strip()}"
        return search_result
    except Exception as e:
        return f"Web search failed: {str(e)}"


# Tool to extract text from a web page using WebBaseLoader
@tool
def web_page_text_extractor(url: str) -> str:
    """
    Extracts and returns plain text content from a given web page URL
    using WebBaseLoader.

    Steps:
      1. Strips whitespace from the URL.
      2. Validates that the URL has a valid HTTP/HTTPS scheme and network location.
      3. Loads the page content via WebBaseLoader.
      4. Returns the concatenated text or a clear error message.

    Args:
        url (str): The URL of the web page to extract content from.

    Returns:
        str: The extracted plain text content, prefixed with the tool name,
             or an error message if validation or loading fails.
    """
    # 1. Clean up the URL
    url_clean = url.strip().lower()

    # 2. Validate URL format
    parsed = urlparse(url_clean)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"[Tool: WebBaseLoader ERROR] Invalid URL: {url_clean}"

    try:
        # 3. Load page content
        loader = WebBaseLoader(url_clean)
        documents = loader.load()
        if not documents:
            return f"[Tool: WebBaseLoader] No content found at {url_clean}"

        # 4. Concatenate and return text
        content = "\n\n".join(doc.page_content for doc in documents)
        return f"[Tool: WebBaseLoader]\n{content}"

    except Exception as e:
        return f"[Tool: WebBaseLoader ERROR] Failed to extract content from {url_clean}: {e}"


class WebSearchTool:
    """DuckDuckGo web search tool enhanced for AI agents workflows"""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
        )

    def search(self, query: str) -> str:
        """Main search method that actually works"""
        try:
            html = self._fetch_search_html(query)
            return self._parse_and_format_results(html)
        except Exception as e:
            return f"Search error: {str(e)}"

    def _fetch_search_html(self, query: str) -> str:
        """Get raw search results HTML"""
        response = self.session.get(
            "https://html.duckduckgo.com/html",
            params={"q": query, "kl": "wt-wt"},
            timeout=10,
        )
        response.raise_for_status()
        return response.text

    def _parse_and_format_results(self, html: str) -> str:
        """Direct parsing without unnecessary filtering"""
        soup = BeautifulSoup(html, "html.parser")
        results = []

        for result in soup.find_all("div", class_="result"):
            try:
                link = result.find("a", class_="result__a")
                snippet = result.find("a", class_="result__snippet")

                if not (link and snippet):
                    continue

                raw_url = link.get("href", "")
                url = self._unwrap_ddg_redirect(raw_url)
                domain = urlparse(url).netloc

                results.append(
                    {
                        "title": link.get_text(strip=True),
                        "url": url,
                        "snippet": snippet.get_text(strip=True),
                        "domain": domain,
                    }
                )

                if len(results) >= self.max_results:
                    break

            except Exception:
                continue

        return self._format_output(results)

    def _unwrap_ddg_redirect(self, ddg_url: str) -> str:
        """Handle DuckDuckGo's URL redirection"""
        if ddg_url.startswith("https://duckduckgo.com/l/"):
            try:
                query = parse_qs(urlparse(ddg_url).query)
                return unquote(query["uddg"][0])
            except:
                return ddg_url
        return ddg_url

    def _format_output(self, results: List[dict]) -> str:
        """Clean markdown formatting"""
        if not results:
            return "No results found."

        output = ["## Search Results"]
        for idx, result in enumerate(results, 1):
            output.append(
                f"{idx}. **[{result['title']}]({result['url']})**\n"
                f"*{result['domain']}*\n"
                f"{result['snippet']}\n"
            )

        return "\n".join(output)
