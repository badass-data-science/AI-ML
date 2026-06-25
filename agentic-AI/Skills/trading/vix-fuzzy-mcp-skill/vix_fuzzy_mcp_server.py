#
# load useful system libraries
#
from __future__ import annotations
import math
from pydantic import BaseModel, ConfigDict
from typing import Dict
from mcp.server.fastmcp import FastMCP

#
# load useful local libraries
#
from python_tools_and_shortcuts.ai.fuzzylogic.FuzzyInterpolator import FuzzyInterpolator
from python_tools_and_shortcuts.econometrics.ticker_prices import get_most_recent_ticker_close_value

#
# user settings
#
list_increasingly_ordered_set_names = ['very low', 'low', 'medium low', 'medium', 'medium high', 'high', 'very high']

dict_membership_ranges = {
    'very low': [9.140000343322754, 12.869999885559082],
    'low': [9.140000343322754, 12.869999885559082, 15.0600004196167],
    'medium low': [12.869999885559082, 15.0600004196167, 17.450000762939453],
    'medium': [15.0600004196167, 17.450000762939453, 20.649999618530273],
    'medium high': [17.450000762939453, 20.649999618530273, 25.110000610351562],
    'high': [20.649999618530273, 25.110000610351562, 82.69000244140625],
    'very high': [25.110000610351562, 82.69000244140625],
}

#
# Initialize an MCP server object
#
mcp = FastMCP("vix-fuzzy-indicator")

#
# define a class for encapsulating the results for MCP
#
class Result(BaseModel):
    """Structured result returned by the MCP tool."""

    model_config = ConfigDict(extra = 'allow')

    #__pydantic_extra__: Dict[str, float]

#
# Define an MCP tool for retrieving VIX and indicating its
# fuzzy class membership that LLMs can discover and execute
#
@mcp.tool()
def get_fuzzy_set_membership_of_most_recent_vix() -> Result:

    """
    Retrieve the most recent VIX indicator close value and specify its
    degree of membership in ordered fuzzy sets describing the VIX value's magnitude.

    Returns:
        A structured fuzzy set result containing a list of ordinal fuzzy set names
        (linguistically corresponding to increasing VIX) and the most recent VIX
        indicator value's degree of membership in each of these sets.

    Notes:
        This tool is informational only and is not financial advice.
    """
    
    vix = get_most_recent_ticker_close_value('^VIX')
    
    # QA
    if not math.isfinite(vix):
        raise RuntimeError(f"Invalid VIX value returned: {vix!r}.")

    # QA
    if vix < 0.0:
        raise RuntimeError(f"Unexpected negative VIX value returned: {vix!r}.")

    fi = FuzzyInterpolator(list_increasingly_ordered_set_names, dict_membership_ranges)
    dict_interpolation = fi.interpolate_membership(vix)

    dict_result = {
        'VIX ordinal increasing fuzzy set names' : list_increasingly_ordered_set_names,
        'VIX fuzzy set membership' : dict_interpolation,
    }
    
    #return Result(**dict_interpolation)
    return Result(**dict_result)

#
# main
#
def main() -> None:
    """Run the MCP server over stdio."""

    #mcp.run(transport="stdio")

    print(get_fuzzy_set_membership_of_most_recent_vix().model_dump_json(indent = 2))


    
#
# entry point
#
if __name__ == '__main__':
    main()
