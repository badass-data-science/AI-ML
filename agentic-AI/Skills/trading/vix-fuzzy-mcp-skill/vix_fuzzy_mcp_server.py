#
# load useful system libraries
#
from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from typing import Dict
from mcp.server.fastmcp import FastMCP

#
# load useful local libraries
#
from vix_fuzzy_shared import VIX_SET_NAMES, get_most_recent_vix, interpolate_vix_membership

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
    
    vix = get_most_recent_vix()

    dict_interpolation = interpolate_vix_membership(vix)

    dict_result = {
        'VIX ordinal increasing fuzzy set names' : VIX_SET_NAMES,
        'VIX fuzzy set membership' : dict_interpolation['fuzzy set membership'],
        'VIX' : vix,
        'VIX value range' : dict_interpolation['value range'],
    }
    
    return Result(**dict_result)

#
# main
#
def main() -> None:
    """Run the MCP server over stdio."""

    #print(get_fuzzy_set_membership_of_most_recent_vix().model_dump_json(indent = 2))
    mcp.run(transport="stdio")
    
#
# entry point
#
if __name__ == '__main__':
    main()
