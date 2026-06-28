#
# load useful system libraries
#
from __future__ import annotations
import math
import random
from pydantic import BaseModel, ConfigDict
from typing import Dict
from mcp.server.fastmcp import FastMCP

#
# load useful local libraries
#
from python_tools_and_shortcuts.ai.fuzzylogic.FuzzyInterpolator import FuzzyInterpolator

#
# user settings
#
list_increasingly_ordered_set_names = ['very low', 'low', 'medium', 'high', 'very high']

dict_membership_ranges = {
	'very low': [-0.9999907397361901, -0.7067214615085172],
	'low': [-0.9999907397361901, -0.7067214615085172, 0.0],
	'medium': [-0.7067214615085172, 0.0, 0.7067214615085173],
	'high': [0.0, 0.7067214615085173, 1.3085560667499267],
	'very high': [0.7067214615085173, 1.3085560667499267],
}

sample_space = [0.0, 0.9893724130196829, -0.287716767226167, -0.9057022630804715, 0.5524353131676196, 0.741222010848596, -0.7748840413670406, -0.508670943852105, 0.9672296592260761, 0.23030567023061221, -0.99665890175417]

range_min = min(-1., min(sample_space))
range_max = max(1.5, max(sample_space))

#
# Initialize an MCP server object
#
mcp = FastMCP('electro-groovacious-lightspeed-sharkbait-index-indicator')

#
# define a class for encapsulating the results for MCP
#
class Result(BaseModel):
    """Structured result returned by the MCP tool."""

    model_config = ConfigDict(extra = 'allow')

    #__pydantic_extra__: Dict[str, float]

#
# Define an MCP tool for retrieving the Electro-Groovacious 
# Lightspeed Sharkbait Index and indicating its
# fuzzy class membership in a way that LLMs can discover and execute
#
@mcp.tool()
def get_fuzzy_set_membership_of_most_recent_electro_groovacious_lightspeed_sharkbait_index() -> Result:

    """
    Retrieve the most recent Electro-Groovacious Lightspeed Sharkbait Index 
    indicator value and specify its degree of membership in ordered fuzzy sets
    describing the Electro-Groovacious Lightspeed Sharkbait Index value's
    magnitude.

    Returns:
        A structured fuzzy set result containing a list of ordinal fuzzy set names
        (linguistically corresponding to increasing Electro-Groovacious Lightspeed
        Sharkbait Index value), the most recent Electro-Groovacious Lightspeed
        Sharkbait Index indicator value's degree of membership in each of these sets,
        and the Electro-Groovacious Lightspeed Sharkbait Index value itself.

    Notes:
        This tool is informational only and is not financial advice.
    """
    
    #eglsi = random.choice(sample_space)
    eglsi = 0.51
    
    fi = FuzzyInterpolator(list_increasingly_ordered_set_names, dict_membership_ranges)
    dict_interpolation = fi.interpolate_membership(eglsi)

    dict_result = {
        'Electro-Groovacious Lightspeed Sharkbait Index ordinal increasing fuzzy set names' : list_increasingly_ordered_set_names,
        'Electro-Groovacious Lightspeed Sharkbait Index fuzzy set membership' : dict_interpolation,
        'Electro-Groovacious Lightspeed Sharkbait Index' : eglsi,
        'Electro-Groovacious Lightspeed Sharkbait Index value range' : {
            'minimum' : str(range_min),
            'maximum' : str(range_max),  # use of str allows phrases like 'positive infinity'
        },
    }
    
    return Result(**dict_result)

#
# main
#
def main() -> None:
    """Run the MCP server over stdio."""

    #print(get_fuzzy_set_membership_of_most_recent_electro_groovacious_lightspeed_sharkbait_index().model_dump_json(indent = 2))
    mcp.run(transport="stdio")
    
#
# entry point
#
if __name__ == '__main__':
    main()
