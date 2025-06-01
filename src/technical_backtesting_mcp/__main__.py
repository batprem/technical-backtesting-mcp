import asyncio

import click
import uvicorn
from fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from technical_backtesting_mcp.backtesters.cross_moving_average import perform_backtest

mcp = FastMCP("Asset Price")


@mcp.tool()
def back_test_asset_price_cross_moving_average(
    symbol: str,
    n1: int,
    n2: int,
    cash: float,
    commission: float,
) -> str:
    """Perform a asset price back testing for a given symbol and return the financial stats.

    The backtesting is done using the cross simple moving average strategy.

    Args:
        symbol (str): TradingView symbol identifier (e.g. "NASDAQ:META", "SET:BH", "BITSTAMP:BTCUSD").
        n1 (int, optional): The first moving average lag. Defaults to 10.
        n2 (int, optional): The second moving average lag. Defaults to 20.
        cash (float, optional): The initial cash. Defaults to 10_000.
        commission (float, optional): The commission. Defaults to 0.002.

    Returns:
        str: The stats of the backtest.
    """
    return perform_backtest(symbol, n1, n2, cash, commission)


async def run_sse_async(mcp: FastMCP, host: str, port: int) -> None:
    """Run the server using SSE transport."""
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0],
                streams[1],
                mcp._mcp_server.create_initialization_options(),
            )

    starlette_app = Starlette(
        debug=mcp.settings.debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    config = uvicorn.Config(
        starlette_app,
        host=host,
        port=port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option("--host", default="0.0.0.0", help="Host to listen on")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(transport: str, host: str, port: int) -> None:
    if transport == "stdio":
        mcp.run(transport=transport)
    elif transport == "sse":
        asyncio.run(run_sse_async(mcp, host, port))


if __name__ == "__main__":
    main()
